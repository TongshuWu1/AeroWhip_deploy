"""Fit only residual weights through independent short coupled rollouts."""

from copy import deepcopy
from time import perf_counter
import math

import numpy as np
import torch

from fitting.batched_objective import BatchedObjective
from fitting.coupled_fit import FitWindow, window_weights, NAMES
from simulation.dynamics.coupled import CoupledDynamics


def check_residual_gradients(objective, network):
    """Check both learned branches on the actual fitting segments, unchanged loss."""
    scales=objective.model.tensor([0.]*len(NAMES))
    def loss():return (objective.rollout(scales)*objective.weights).sum()
    network.zero_grad(set_to_none=True)
    value=loss();value.backward()
    branches=[]
    for name, branch in (('drone',network.drone),('cable',network.cable)):
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
                agrees=math.isfinite(finite) and abs(finite-analytic)<=1e-4+.02*max(abs(finite),abs(analytic))
                checks.append({'epsilon':epsilon,'finite_difference':finite,'agrees':agrees})
        finally:
            with torch.no_grad():
                for p,v in zip(parameters,original):p.copy_(v)
        a,b=(c['finite_difference'] for c in checks)
        stable=math.isfinite(a) and math.isfinite(b) and abs(a-b)<=1e-4+.02*max(abs(a),abs(b))
        branches.append({'branch':name,'analytic':analytic,'checks':checks,
                         'passed':stable and all(c['agrees'] for c in checks)})
    network.zero_grad(set_to_none=True)
    return {'loss':float(value.detach()),'passed':all(b['passed'] for b in branches),
            'absolute_tolerance':1e-4,'relative_tolerance':.02,'branches':branches}


def fit_residual(config, windows, network, settings, *, progress=None, checkpoint=None):
    integers = ('maximum_updates','minimum_updates','check_every','patience_checks')
    if any(not isinstance(settings[k],int) or settings[k]<1 for k in integers):
        raise ValueError('Positive integer training intervals required')
    if settings['minimum_updates']>settings['maximum_updates']:
        raise ValueError('Minimum updates exceeds maximum updates')
    for key in ('learning_rate','magnitude_penalty','gradient_clip','relative_improvement'):
        if not math.isfinite(settings[key]) or settings[key]<=0:
            raise ValueError('Positive finite training setting required: '+key)
    started=perf_counter()
    network.train().requires_grad_(True)
    device=str(next(network.parameters()).device)
    if progress:progress({'phase':'capture'})
    objective=BatchedObjective(config,windows,window_weights(windows),device,
        graph=device.startswith('cuda'),compile_physics=device.startswith('cuda'),
        residual=network,magnitude_penalty=settings['magnitude_penalty'])
    scales=objective.model.tensor([0.]*len(NAMES))
    optimizer=torch.optim.Adam(network.parameters(),lr=settings['learning_rate'])

    def loss():
        return (objective.rollout(scales)*objective.weights).sum()

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
        value=loss()
        if not bool(torch.isfinite(value)):raise FloatingPointError('Nonfinite residual training loss')
        value.backward()
        if any(p.grad is None for p in network.parameters()):
            raise RuntimeError('Missing residual parameter gradient')
        norm=torch.nn.utils.clip_grad_norm_(network.parameters(),settings['gradient_clip'],error_if_nonfinite=True)
        optimizer.step()
        row={'update':update,'loss_before_update':float(value.detach()),
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
    network.eval().requires_grad_(False)
    return {'training_loss_before':baseline,'training_loss_after':best_loss,
            'selected_update':best_update,'updates':update,'stop_reason':reason,
            'history':history,'elapsed_s':perf_counter()-started,
            'gradient_checks':{'initial':before_check,'selected':after_check},
            'training_segments':len(windows),'segment_steps':len(windows[0].applied_commands),
            'physical_parameters_frozen':True,'promoted':False}


def hold_comparison(config, network, duration_s=30.):
    """Stationary-start diagnostic, not a proof of long-horizon stability."""
    model=CoupledDynamics(config)
    initial=model.initial_state()
    p,_=model.tracking_pose(initial);q,_=model.cable_state(initial)
    steps=round(duration_s/model.dt)
    commands=np.zeros((steps,11));commands[:,:3]=p.numpy()
    window=FitWindow('stationary_hold',initial,commands,
        np.broadcast_to(p.numpy(),(steps,3)).copy(),
        np.broadcast_to(initial.rotation.numpy(),(steps,3,3)).copy(),
        np.broadcast_to(q[model.marker_indices[1:]].numpy(),(steps,len(model.marker_indices)-1,3)).copy())
    probe=deepcopy(network).eval().requires_grad_(False)
    selected=deepcopy(probe.state_dict())
    with torch.no_grad():
        for parameter in probe.parameters():parameter.zero_()
    batch=BatchedObjective(config,[window],[1.],str(next(probe.parameters()).device),
        graph=next(probe.parameters()).is_cuda,training=False,residual=probe)
    metrics={};arrays={'time_s':np.arange(1,steps+1)*model.dt}
    for label in ('before','after'):
        if label=='after':probe.load_state_dict(selected)
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
