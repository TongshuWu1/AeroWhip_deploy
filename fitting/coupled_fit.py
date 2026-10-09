"""Bounded fitting of the coupled model from explicitly prepared flight windows.

No raw-file discovery, clock alignment, future-state initialization, or automatic
model promotion. Commands must already include the chosen command delay; target
row k is measured after command k's 10 ms interval. Separate recording groups
are required for training and validation. Results are returned, not written.
"""

from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, fields

import numpy as np
import torch
from torch.utils.checkpoint import checkpoint

from simulation.tensor_model import TensorModel, PlantState, ControlledState
from controllers.full_state_controller.tensor_controller import GeometricController


@dataclass(frozen=True)
class FitWindow:
    recording_group: str
    initial: PlantState
    applied_commands: np.ndarray
    tracking_positions: np.ndarray
    tracking_rotations: np.ndarray
    cable_markers: np.ndarray
    timestep_s: float = .01
    maneuver: str = "unspecified"

    def validate(self, model):
        n = len(self.applied_commands)
        width = np.shape(self.applied_commands)[-1]
        if width not in (11,13):
            raise ValueError('Expected legacy zero-yaw-rate or full three-body-rate commands')
        expected = ((self.applied_commands,(n,width)),(self.tracking_positions,(n,3)),
                    (self.tracking_rotations,(n,3,3)),
                    (self.cable_markers,(n,len(model.marker_indices)-1,3)))
        if not self.recording_group or n<1 or abs(self.timestep_s-model.dt)>1e-12:
            raise ValueError("Windows need a recording group, samples and matching timestep")
        for value,shape in expected:
            if np.shape(value)!=shape or not np.isfinite(value).all():
                raise ValueError(f"Expected complete finite window array {shape}; split at data gaps")
        if width == 11 and np.any(self.applied_commands[:,10]!=0):
            raise ValueError('Legacy yaw-rate packets must be zero; use three desired body rates')
        R=self.tracking_rotations
        if not np.allclose(np.swapaxes(R,-1,-2)@R,np.eye(3),atol=1e-6,rtol=0) or not np.allclose(np.linalg.det(R),1,atol=1e-6,rtol=0):
            raise ValueError("Target orientations must be proper rotations")
        for field in fields(self.initial):
            value=getattr(self.initial,field.name)
            if value.device.type!='cpu' or not bool(torch.isfinite(value).all()):
                raise ValueError("This first fitter expects finite CPU initial states")
        q,_=model.cable_state(self.initial)
        if float(((q[1:]-q[:-1]).norm(dim=1)-model.rest).abs().max())>1e-6:
            raise ValueError("Initial cable state must satisfy the configured lengths")


NAMES=("position_gain", "velocity_gain", "attitude_frequency")
# Common XYZ gain scales preserve axis ratios and limit the number of unknowns.
# Bending stiffness, geometry, masses, inertia, motors and clocks remain fixed.
BOUNDS=((.5,2.),(.5,2.),(.5,3.))


def _scaled_config(base, scales):
    if len(scales) != len(NAMES):
        raise ValueError(f'Expected {len(NAMES)} scales in order {NAMES}')
    config=deepcopy(base)
    control=config['controller']
    for key,scale in (("position_gain_s_inv2",scales[0]),("velocity_gain_s_inv",scales[1]),
                      ("attitude_frequency_rad_s",scales[2])):
        control[key]=[float(v*scale) for v in control[key]]
    # All cable coefficients are owned by measured-root identification.
    # Retain the coupled plant and its reactions; never retune cable drag here.
    return config


def window_loss(base, window, log_scales):
    """One full untruncated rollout; checkpointing trades recomputation for RAM."""
    if log_scales.shape != (len(NAMES),):
        raise ValueError(f'Expected {len(NAMES)} log-scales in order {NAMES}')
    scales=log_scales.exp()
    model=TensorModel(base);controller=GeometricController(model)
    controller.position_gain_s_inv2=controller.position_gain_s_inv2*scales[0]
    controller.velocity_gain_s_inv=controller.velocity_gain_s_inv*scales[1]
    controller.attitude_frequency_rad_s=controller.attitude_frequency_rad_s*scales[2]
    state=model.controller_state(PlantState(*(getattr(window.initial,f.name).detach().clone() for f in fields(window.initial))))
    commands=model.tensor(window.applied_commands)
    target_positions=model.tensor(window.tracking_positions)
    target_rotations=model.tensor(window.tracking_rotations)
    loss=model.tensor(0.)
    def advance(*args):
        output=model.step(ControlledState(*args[:-1]),args[-1],controller)
        return tuple(getattr(output,f.name) for f in fields(output))
    for k,command in enumerate(commands):
        if torch.is_grad_enabled():
            tensors=tuple(getattr(state,f.name) for f in fields(state))
            state=ControlledState(*checkpoint(advance,*tensors,command,use_reentrant=False))
        else:
            state=model.step(state,command,controller)
        p,_=model.tracking_pose(state)
        loss=loss+((p-target_positions[k])/.02).square().mean()
        loss=loss+((state.rotation-target_rotations[k])/.05).square().sum()/2
    return loss/len(window.applied_commands)


def _evaluate_window(args):
    base,window,values,backward=args
    torch.set_num_threads(1)
    x=torch.tensor(values,dtype=torch.float64,requires_grad=backward)
    with torch.set_grad_enabled(backward):
        loss=window_loss(base,window,x)
        if not bool(torch.isfinite(loss)):raise FloatingPointError("Nonfinite fitting loss")
        if backward:
            loss.backward()
            if x.grad is None or not bool(torch.isfinite(x.grad).all()):
                raise FloatingPointError("Nonfinite fitting gradient")
    return float(loss.detach()), x.grad.tolist() if backward else None


def window_weights(windows):
    """Equal motion types, equal sessions within each type, then equal takes."""
    group_types = {}
    for window in windows:
        previous = group_types.setdefault(window.recording_group, window.maneuver)
        if previous != window.maneuver:
            raise ValueError("A recording group cannot belong to multiple motion types")
    types = set(group_types.values())
    return [1 / (len(types)
                 * sum(value == w.maneuver for value in group_types.values())
                 * sum(v.recording_group == w.recording_group for v in windows))
            for w in windows]


def fit_coupled(base_config, training, validation, *, iterations=30, learning_rate=.05, progress=None, workers=4, device='cpu', compile_physics=True):
    if base_config.get('residual',{}).get('status') != 'unfitted':
        raise ValueError('Calibrate physics before aerodynamic stages; do not compensate a learned drag law by refitting physics')
    if workers<1:raise ValueError("At least one worker is required")
    if device not in ('native','cpu','cuda'):raise ValueError("Fitting execution must be native, cpu or cuda")
    if device == 'cuda':
        if not torch.cuda.is_available():raise RuntimeError("CUDA is not available")
        return _fit_coupled(base_config,training,validation,iterations,learning_rate,progress,None,device,compile_physics)
    # Processes evaluate independent windows, never different scientific models.
    if workers>1:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            return _fit_coupled(base_config,training,validation,iterations,learning_rate,progress,executor,device)
    return _fit_coupled(base_config,training,validation,iterations,learning_rate,progress,None,device)


def _fit_coupled(base_config, training, validation, iterations, learning_rate, progress, executor, device='cpu', compile_physics=False):
    """Return a candidate and before/after held-out losses; never install it.

    Motion types have equal weight, then recording groups within each type.
    Loss balances vehicle tracking position (2 cm) and rotation
    matrix error (0.05 rad small-angle scale). These are objective scales, not
    acceptance tolerances. No validation values enter optimization.
    """
    if not training or not validation or iterations<1 or learning_rate<=0:
        raise ValueError("Provide separate training/validation windows and positive fit settings")
    if {w.recording_group for w in training}&{w.recording_group for w in validation}:
        raise ValueError("Training and validation recording groups overlap")
    base=deepcopy(base_config)
    model=TensorModel(base)
    for window in (*training,*validation):window.validate(model)
    log_scale=torch.zeros(len(NAMES),dtype=torch.float64,requires_grad=True)
    lower=torch.log(torch.tensor([v[0] for v in BOUNDS],dtype=torch.float64))
    upper=torch.log(torch.tensor([v[1] for v in BOUNDS],dtype=torch.float64))
    optimizer=torch.optim.Adam([log_scale],lr=learning_rate)
    batches = {}
    if device == 'cuda':
        from fitting.batched_objective import BatchedObjective
        torch.backends.cuda.preferred_linalg_library('cusolver')
        for windows in (training, validation):
            if progress:progress({'phase':'preparing_gpu',
                                  'role':'fitting' if windows is training else 'comparison',
                                  'windows':len(windows)})
            batches[id(windows)] = BatchedObjective(base, windows, window_weights(windows), device,
                graph=True, compile_physics=compile_physics and windows is training,
                training=windows is training, loss_scope='vehicle')

    def objective(windows,values,*,backward=False):
        if batches:
            total, gradient = batches[id(windows)].evaluate(values.detach().tolist(), backward)
            if backward:
                values.grad = gradient if values.grad is None else values.grad + gradient
            if progress:progress({'phase':'gradient' if backward else 'evaluation',
                                  'window':len(windows),'windows':len(windows),'device':device})
            return total
        total=0.
        jobs=[(base,w,values.detach().tolist(),backward) for w in windows]
        if device == 'native':
            from fitting.native_objective import evaluate_window as evaluate
        else:
            evaluate = _evaluate_window
        outputs=executor.map(evaluate,jobs) if executor else map(evaluate,jobs)
        for index,(window,weight,(loss,gradient)) in enumerate(zip(windows,window_weights(windows),outputs)):
            total+=weight*loss
            if backward:
                if values.grad is None:values.grad=torch.zeros_like(values)
                values.grad+=values.new_tensor(gradient)*weight
            if progress:progress({"phase":"gradient" if backward else "evaluation", "window":index+1,
                                  "windows":len(windows),"recording_group":window.recording_group})
        return total

    derivative_checks = {}
    def check_derivatives(values, stage):
        if device != 'native':
            return
        from fitting.native_objective import check_window
        jobs = [(base,w,values.detach().tolist()) for w in training]
        outputs = executor.map(check_window,jobs) if executor else map(check_window,jobs)
        rows = []
        for row in outputs:
            rows.append(row)
            if progress:progress(dict(phase='physical_gradient_check',stage=stage,**row))
            if not row['passed']:
                raise RuntimeError('Native finite-difference step-size check failed; candidate not accepted')
        derivative_checks[stage] = rows

    check_derivatives(log_scale, 'initial')
    if progress:progress({'phase':'baseline','windows':len(training)})
    with torch.no_grad():
        before_train=objective(training,log_scale)
    if progress:progress({'phase':'baseline_complete','training_loss':before_train})
    best=log_scale.detach().clone();best_loss=before_train;history=[]
    for iteration in range(iterations):
        optimizer.zero_grad();current=objective(training,log_scale,backward=True)
        if log_scale.grad is None or not bool(torch.isfinite(log_scale.grad).all()):
            raise FloatingPointError("Nonfinite fitting gradient")
        history.append(current)
        if current<best_loss:best_loss=current;best=log_scale.detach().clone()
        optimizer.step()
        with torch.no_grad():log_scale.clamp_(lower,upper)
        if progress:progress({"phase":"iteration","iteration":iteration+1,"iterations":iterations,
                              "training_loss":current,"best_training_loss":best_loss,
                              "best_scales":dict(zip(NAMES,best.exp().tolist())),
                              "best_config":_scaled_config(base,best.exp().tolist())})
    with torch.no_grad():
        last=objective(training,log_scale)
        if last<best_loss:best_loss=last;best=log_scale.detach().clone()
        check_derivatives(best, 'selected')
        # Comparison data are evaluated only after training has selected its candidate.
        before_validation=objective(validation,log_scale.new_zeros(len(NAMES)))
        after_validation=objective(validation,best)
    return {"config":_scaled_config(base,best.exp().tolist()),
            "training_loss_before":before_train,"training_loss_after":best_loss,
            "validation_loss_before":before_validation,"validation_loss_after":after_validation,
            "scales":dict(zip(NAMES,best.exp().tolist())),"training_history":history,
            "last_training_loss":last,"iterations":iterations,"converged":False,
            "promoted":False,"device":device,"compiled":device=='cuda' and compile_physics,
            "loss_scope":"vehicle position and attitude; full coupled dynamics; cable fixed",
            "derivatives":"central finite differences in three log scales" if device=='native' else "automatic differentiation",
            "derivative_checks":derivative_checks}
