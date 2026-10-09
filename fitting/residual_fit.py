"""Fit aerodynamic branches sequentially with physically owned observation losses."""

from copy import deepcopy
from contextlib import ExitStack
from time import perf_counter
import math

import numpy as np
import torch

from fitting.batched_objective import BatchedObjective
from fitting.coupled_fit import FitWindow, window_weights, NAMES
from simulation.tensor_model import TensorModel
from fitting.prescribed_residual_objective import PrescribedResidualObjective


class ResidualObjective:
    """Memory-bounded vehicle-only objective with the cable branch frozen.

    Each window starts independently. All windows and original global weights
    participate in every loss/gradient evaluation. Gradients accumulate before
    one global clipping operation and one optimizer step. No stochastic
    minibatching, per-chunk normalization, padding or altered integration.
    """
    def __init__(self, config, windows, network, magnitude_penalty, batch_size, progress=None):
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
            raise ValueError('Execution batch size must be a positive integer')
        if not windows:
            raise ValueError('Residual fitting needs independent windows')
        if network.fitting_stage != 'drone' or any(p.requires_grad for p in network.cable.parameters()):
            raise ValueError('Vehicle residual loss requires a selected drone stage and frozen cable network')
        self.config, self.windows, self.network = config, windows, network
        self.weights = window_weights(windows)
        self.batch_size = min(batch_size, len(windows))
        self.magnitude_penalty, self.progress = magnitude_penalty, progress
        self.device = str(next(network.parameters()).device)
        self.objectives = {}
        self.passes = 0

    def loss(self, backward=False):
        total = 0.
        self.passes += 1
        for begin in range(0, len(self.windows), self.batch_size):
            end = min(begin+self.batch_size, len(self.windows))
            windows, weights = self.windows[begin:end], self.weights[begin:end]
            size = end-begin
            if size not in self.objectives:
                if self.progress:
                    self.progress(dict(phase='capture',execution_batch_size=size,total_segments=len(self.windows)))
                self.objectives[size] = BatchedObjective(self.config, windows, weights, self.device,
                    graph=self.device.startswith('cuda'), compile_physics=self.device.startswith('cuda'),
                    residual=self.network, magnitude_penalty=self.magnitude_penalty, loss_scope='vehicle')
            objective = self.objectives[size]
            objective.set_windows(windows, weights)
            scales = objective.model.tensor([0.]*len(NAMES))
            with torch.set_grad_enabled(backward):
                value = (objective.rollout(scales)*objective.weights).sum()
                if not bool(torch.isfinite(value)):
                    raise FloatingPointError('Nonfinite residual loss in execution chunk')
                if backward:
                    value.backward()
                total += float(value.detach())
            if self.progress:
                self.progress(dict(phase='loss_pass',loss_pass=self.passes,backward=backward,
                    completed_segments=end,total_segments=len(self.windows),execution_batch_size=size))
        return total


def _weighted_loss(objective, backward=False):
    if isinstance(objective, (ResidualObjective,PrescribedResidualObjective)):
        return objective.loss(backward)
    # Retain the direct objective interface for bounded gradient/parity checks.
    with torch.set_grad_enabled(backward):
        scales=objective.model.tensor([0.]*len(NAMES))
        value=(objective.rollout(scales)*objective.weights).sum()
        if backward:value.backward()
        return float(value.detach())


def check_residual_gradients(objective, network):
    """Finite differences of the selected branch on its actual identification loss."""
    def loss(backward=False):return _weighted_loss(objective,backward)
    network.zero_grad(set_to_none=True)
    value=loss(backward=True)
    branches=[]
    for name in (network.fitting_stage,):
        branch = getattr(network, name)
        parameters=list(branch.parameters())
        if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in parameters):
            raise FloatingPointError('Missing/nonfinite residual gradient: '+name)
        gradients=[p.grad.detach().clone() for p in parameters]
        norm=torch.sqrt(sum(g.square().sum() for g in gradients))
        if float(norm)==0:
            raise FloatingPointError('Cannot verify a zero residual branch gradient: '+name)
        direction=[g/norm for g in gradients]
        original=[p.detach().clone() for p in parameters]
        analytic=float(norm);checks=[]
        try:
            for epsilon in (1e-5,5e-6):
                with torch.no_grad():
                    for p,v,d in zip(parameters,original,direction):p.copy_(v+epsilon*d)
                    plus=float(loss())
                    for p,v,d in zip(parameters,original,direction):p.copy_(v-epsilon*d)
                    minus=float(loss())
                finite=(plus-minus)/(2*epsilon)
                agrees=math.isfinite(finite) and abs(finite-analytic)<=1e-8+.02*max(abs(finite),abs(analytic))
                checks.append({'epsilon':epsilon,'finite_difference':finite,'agrees':agrees})
        finally:
            with torch.no_grad():
                for p,v in zip(parameters,original):p.copy_(v)
        a,b=(c['finite_difference'] for c in checks)
        stable=math.isfinite(a) and math.isfinite(b) and abs(a-b)<=1e-8+.02*max(abs(a),abs(b))
        branches.append({'branch':name,'analytic':analytic,'checks':checks,
                         'passed':stable and all(c['agrees'] for c in checks)})
    network.zero_grad(set_to_none=True)
    return {'loss':value,'passed':all(b['passed'] for b in branches),
            'absolute_tolerance':1e-8,'relative_tolerance':.02,'branches':branches}


def fit_residual(config, windows, network, settings, *, stage, progress=None, checkpoint=None, cable_workers=1):
    """Fit one owned branch; optional CPU workers preserve the full objective."""
    if not isinstance(cable_workers,int) or isinstance(cable_workers,bool) or cable_workers < 1:
        raise ValueError('Positive integer cable worker count required')
    if stage != 'cable' and cable_workers != 1:
        raise ValueError('CPU cable workers cannot change drone-stage execution')
    with ExitStack() as stack:
        return _fit_residual(config,windows,network,settings,stage=stage,progress=progress,
                             checkpoint=checkpoint,cable_workers=cable_workers,stack=stack)


def _fit_residual(config, windows, network, settings, *, stage, progress, checkpoint, cable_workers, stack):
    integers = ('maximum_updates','minimum_updates','check_every','patience_checks')
    if any(not isinstance(settings[k],int) or settings[k]<1 for k in integers):
        raise ValueError('Positive integer training intervals required')
    if settings['minimum_updates']>settings['maximum_updates']:
        raise ValueError('Minimum updates exceeds maximum updates')
    for key in ('learning_rate','magnitude_penalty','gradient_clip','relative_improvement'):
        if not math.isfinite(settings[key]) or settings[key]<=0:
            raise ValueError('Positive finite training setting required: '+key)
    started=perf_counter()
    network.train()
    network.select_stage(stage)
    frozen = {k:v.detach().clone() for k,v in network.state_dict().items() if not k.startswith(stage+'.')}
    device=str(next(network.parameters()).device)
    if stage == 'cable' and cable_workers > 1:
        from fitting.parallel_prescribed_objective import ParallelPrescribedResidualObjective
        objective=stack.enter_context(ParallelPrescribedResidualObjective(config,windows,network,
            settings['magnitude_penalty'],cable_workers,progress))
    else:
        objective=(PrescribedResidualObjective(config,windows,network,settings['magnitude_penalty'],progress)
                   if stage=='cable' else ResidualObjective(config,windows,network,settings['magnitude_penalty'],
                        settings['execution_batch_size'],progress))
    parameters = list(getattr(network,stage).parameters())
    optimizer=torch.optim.Adam(parameters,lr=settings['learning_rate'])

    def loss(backward=False):
        return objective.loss(backward)

    before_check=check_residual_gradients(objective,network)
    if progress:progress({'phase':'gradient_check','stage':'initial',**before_check})
    if not before_check['passed']:
        raise RuntimeError('Initial short-segment gradients failed finite differences; fit not started')
    with torch.no_grad():baseline=float(loss())
    if not math.isfinite(baseline):raise FloatingPointError('Nonfinite baseline residual loss')
    best_loss=baseline;best=deepcopy(network.state_dict());best_update=0
    significant=baseline;stale=0;history=[];reason='update_limit'
    if progress:progress({'phase':'baseline','loss':baseline,'elapsed_s':perf_counter()-started})
    for update in range(1,settings['maximum_updates']+1):
        optimizer.zero_grad(set_to_none=True)
        value=loss(backward=True)
        if not math.isfinite(value):raise FloatingPointError('Nonfinite residual training loss')
        if any(p.grad is None for p in parameters):
            raise RuntimeError('Missing residual parameter gradient')
        norm=torch.nn.utils.clip_grad_norm_(parameters,settings['gradient_clip'],error_if_nonfinite=True)
        optimizer.step()
        row={'update':update,'loss_before_update':value,
             'gradient_norm':float(norm),'elapsed_s':perf_counter()-started}
        if update%settings['check_every']==0 or update==settings['maximum_updates']:
            with torch.no_grad():score=float(loss())
            if not math.isfinite(score):raise FloatingPointError('Nonfinite updated residual loss')
            if score<best_loss:
                best_loss=score;best=deepcopy(network.state_dict());best_update=update
            if score<significant*(1-settings['relative_improvement']):
                significant=score;stale=0
            else:stale+=1
            row.update(selection_loss=score,best_loss=best_loss,selected_update=best_update)
        history.append(row)
        if progress:progress({'phase':'training',**row})
        if checkpoint and 'selection_loss' in row:
            checkpoint({'current':network.state_dict(),'best':best,'optimizer':optimizer.state_dict(),
                        'update':update,'selected_update':best_update,'history':history})
        if update>=settings['minimum_updates'] and stale>=settings['patience_checks']:
            reason='training_plateau';break
    network.load_state_dict(best)
    after_check=check_residual_gradients(objective,network)
    if progress:progress({'phase':'gradient_check','stage':'selected',**after_check})
    if not after_check['passed']:
        raise RuntimeError('Selected short-segment gradients failed finite differences; candidate not exported')
    if any(not torch.equal(network.state_dict()[k],v) for k,v in frozen.items()):
        raise RuntimeError('Frozen residual branch changed during fitting')
    network.completed_stages = ['cable'] if stage=='cable' else ['cable','drone']
    network.eval().requires_grad_(False)
    return {'stage':stage,'training_loss_before':baseline,'training_loss_after':best_loss,
            'loss_scope':'measured_support_cable_markers' if stage=='cable' else 'coupled_vehicle_pose_only',
            'frozen_branch_unchanged':True,
            'selected_update':best_update,'updates':update,'stop_reason':reason,
            'history':history,'elapsed_s':perf_counter()-started,
            'gradient_checks':{'initial':before_check,'selected':after_check},
            'training_segments':len(windows),'segment_steps':len(windows[0].applied_commands),
            'execution_batch_size':objective.batch_size,
            'execution_workers':getattr(objective,'workers',1),
            'physical_parameters_frozen':True,'promoted':False}


def hold_comparison(config, network, duration_s=30.):
    """Stationary-start diagnostic, not a proof of long-horizon stability."""
    model=TensorModel(config)
    initial=model.initial_state()
    p,_=model.tracking_pose(initial);q,_=model.cable_state(initial)
    steps=round(duration_s/model.dt)
    commands=np.zeros((steps,11));commands[:,:3]=p.numpy()
    window=FitWindow('stationary_hold',initial,commands,
        np.broadcast_to(p.numpy(),(steps,3)).copy(),
        np.broadcast_to(initial.rotation.numpy(),(steps,3,3)).copy(),
        np.broadcast_to(q[model.marker_indices[1:]].numpy(),(steps,len(model.marker_indices)-1,3)).copy())
    metrics={};arrays={'time_s':np.arange(1,steps+1)*model.dt}
    for label,residual in (('before',None),('after',network)):
        batch=BatchedObjective(config,[window],[1.],str(next(network.parameters()).device),
            graph=next(network.parameters()).is_cuda,training=False,residual=residual)
        positions,markers,rotations=(a[0] for a in batch.predict([0.]*len(NAMES)))
        speed=np.linalg.norm(np.diff(markers[:,-1],axis=0)/model.dt,axis=1)
        tail=round(5/model.dt)
        error=np.linalg.norm(positions-p.numpy(),axis=1)
        metrics[label]={'final_position_error_m':float(error[-1]),'peak_position_error_m':float(error.max()),
            'tip_speed_rms_last_5s_m_s':float(np.sqrt(np.mean(speed[-tail:]**2))),
            'tip_speed_rms_previous_5s_m_s':float(np.sqrt(np.mean(speed[-2*tail:-tail]**2))),
            'tip_span_last_5s_m':float(np.linalg.norm(np.ptp(markers[-tail:,-1],axis=0)))}
        arrays.update({label+'_positions':positions,label+'_markers':markers,label+'_rotations':rotations})
    return {'duration_s':duration_s,'metrics':metrics},arrays
