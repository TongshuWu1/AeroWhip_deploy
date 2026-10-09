"""Recorded component/end-to-end comparisons; no optimizer or model updates."""

from copy import deepcopy
from dataclasses import fields, replace
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import torch

from fitting.recorded_replay import prepare_recorded_replay
from project_paths import ROOT, project_path, project_reference
from simulation.configuration import (InitialStateConfiguration, PlatformConfiguration,
                                      load_configuration)
from simulation.tensor_model import TensorModel
from simulation.model_configuration import load_model_configuration, verify_configuration_sources
from simulation.native_runtime import NativeRuntime
from simulation.prescribed_cable import PrescribedCable, PrescribedCableState, NativePrescribedCable
from simulation.state import rotation_matrix_to_quaternion_xyzw
from simulation.timing import TIME_EPS
from take_processing.library import verify_sources
from take_processing.take_data import digest


def _numpy(value):
    return (value.detach().cpu().numpy().copy() if isinstance(value, torch.Tensor)
            else np.array(value, copy=True))


def _runtime(model, replay):
    """Initialize once from causal history, then let the coupled plant evolve."""
    initial = replay.initial
    p, v = model.tracking_pose(initial)
    q, qv = model.cable_state(initial)
    scene = replace(load_configuration('configs/scenes/default.toml'),
        timestep_s=model.dt,
        platform=PlatformConfiguration(_numpy(model.attachment_tracking),
            np.array(model.config['cable']['marker_interval_lengths_m'])),
        initial_state=InitialStateConfiguration(_numpy(p),
            rotation_matrix_to_quaternion_xyzw(_numpy(initial.rotation)),
            _numpy(v), _numpy(initial.omega_body), _numpy(q), _numpy(qv), _numpy(initial.material_angles)))
    # Replay command timestamps already include transport delay. Applying the
    # model delay a second time here would change the input experiment.
    runtime = NativeRuntime(scene, model_config=model.config, device='cpu', command_delay_s=0.)
    for field in fields(initial):
        if not np.allclose(_numpy(getattr(initial, field.name)), getattr(runtime.physics, field.name),
                          atol=1e-12, rtol=0):
            raise ValueError('Runtime initialization changed the causal recorded state')
    return runtime


def _identities(model_path, config, configuration_sources=None):
    from simulation import _native, _tensor_native
    if configuration_sources is None:
        resolved,configuration_sources=load_model_configuration(model_path,with_sources=True)
        if resolved != config:
            raise RuntimeError('Model changed while loading comparison inputs')
    paths = {'model': model_path}
    paths['native_binary'] = Path(_native.__file__)
    paths['tensor_binary'] = Path(_tensor_native.__file__)
    residual = config.get('residual', {})
    if residual.get('status') == 'fitted':
        paths['residual'] = project_path(residual['checkpoint'])
    for relative in ('fitting/calibration.py', 'fitting/recorded_replay.py',
                     'fitting/prepare_window.py', 'take_processing/take_data.py',
                     'simulation/configuration.py', 'simulation/state.py',
                     'simulation/timing.py', 'simulation/tensor_model.py', 'simulation/model.py',
                     'simulation/model_configuration.py',
                     'controllers/full_state_controller/tensor_controller.py',
                     'controllers/full_state_controller/geometric_controller.py', 'simulation/actuator.py',
                     'native/actuator.cpp', 'native/actuator.hpp',
                     'simulation/native_runtime.py', 'simulation/prescribed_cable.py',
                     'fitting/residual_network.py', 'native/engine.cpp', 'native/engine.hpp',
                     'native/runtime.cpp', 'native/runtime.hpp', 'native/bindings.cpp',
                     'native/tensor_physics.cpp', 'native/material_frame.inc',
                     'native/tensor_material.inc', 'native/prescribed_material.inc'):
        paths[relative] = ROOT / relative
    identities = {key: {'path': project_reference(path), 'sha256': digest(path)} for key, path in paths.items()}
    identities.update(configuration_sources)
    verify_configuration_sources(identities)
    return identities


@torch.no_grad()
def compare_recording(path, model_path, *, start_s, duration_s, mode='prescribed_cable',
                      progress=None, cancelled=None, device='cpu'):
    """Return a saved-replay-compatible diagnostic, without writing any files.

    Cable mode prescribes measured root motion and retains cable aerodynamic corrections.
    Coupled mode uses native recorded commands and never injects later measured
    states. The canonical sampled geometric controller and motors are used.
    """
    if mode not in ('prescribed_cable', 'coupled'):
        raise ValueError('Choose prescribed_cable or coupled comparison')
    model_path = project_path(model_path)
    config,configuration_sources = load_model_configuration(model_path,with_sources=True)
    identities = _identities(model_path, config, configuration_sources)
    if device == 'cpu':
        torch.set_num_threads(1)
    model = TensorModel(config, device=device)
    cable = (NativePrescribedCable(model) if device == 'cpu' else PrescribedCable(model)) if mode == 'prescribed_cable' else None

    def check_cancelled():
        if cancelled is not None and cancelled():
            raise InterruptedError('Comparison stopped; no result saved')

    check_cancelled()
    replay = prepare_recorded_replay(path, model, start_s=start_s, duration_s=duration_s,
                                    require_commands=cable is None)
    return _compare_prepared(model, replay, identities, mode=mode,
                             progress=progress, cancelled=cancelled)


@torch.no_grad()
def _compare_prepared(model, replay, identities, *, mode='prescribed_cable',
                      progress=None, cancelled=None):
    """Shared replay kernel. Cached preparation requires unchanged initialization physics."""
    config, device = model.config, model.device
    cable = (NativePrescribedCable(model) if device == 'cpu' else PrescribedCable(model)) if mode == 'prescribed_cable' else None

    def check_cancelled():
        if cancelled is not None and cancelled():
            raise InterruptedError('Comparison stopped; no result saved')

    check_cancelled()
    verify_sources({'sources': replay.metadata['sources']})
    runtime = _runtime(model, replay) if cable is None else None
    initial = runtime.physics if runtime is not None else replay.initial
    arrays = {'initial_'+field.name: _numpy(getattr(initial, field.name)) for field in fields(initial)}
    state = PrescribedCableState(initial.free_positions, initial.free_velocities, initial.material_angles) if cable is not None else None
    positions, rotations, nodes, velocities, reactions, moments, ticks = [], [], [], [], [], [], []
    material_angles = []
    max_length_error = max_rate_error = 0.
    event_index = 0
    total = len(replay.time_s)-1
    if progress is not None:
        progress(0, total)
    for i in range(total):
        check_cancelled()
        if cable is not None:
            start, end = map(float, replay.time_s[i:i+2])
            state = cable.step(state, start, end-start, replay.attachment_motion)
            boundary = replay.attachment_motion(end)
            q = torch.cat((boundary[0][None], state.free_positions))
            v = torch.cat((boundary[1][None], state.free_velocities))
            _, reaction, moment = cable.acceleration(state, boundary)
            reactions.append(_numpy(reaction))
            moments.append(_numpy(moment))
            material_angles.append(_numpy(state.material_angles))
            p, R = replay.tracking_positions[i+1], replay.tracking_rotations[i+1]
        else:
            # Match the runtime's right-continuous native event convention;
            # an event at an output endpoint belongs to the next interval.
            events = []
            while (event_index < len(replay.command_times_s)
                   and replay.command_times_s[event_index] < (i+1)*model.dt-TIME_EPS):
                events.append((replay.command_times_s[event_index], replay.commands[event_index]))
                event_index += 1
            runtime.step(command_events=events)
            physical = runtime.physics
            material_angles.append(_numpy(physical.material_angles))
            if not all(np.isfinite(getattr(physical, field.name)).all() for field in fields(physical)):
                raise FloatingPointError('Nonfinite coupled replay state')
            q, v = runtime.model.cable_state(physical)
            p = _numpy(runtime.model.tracking_pose(physical)[0])
            R = _numpy(physical.rotation)
            ticks.extend(runtime.last_controller_update_times)
        q, v = _numpy(q), _numpy(v)
        if not all(np.isfinite(value).all() for value in (p, R, q, v)):
            raise FloatingPointError('Nonfinite comparison output')
        if (not np.allclose(R.T @ R, np.eye(3), rtol=0, atol=1e-6)
                or not np.isclose(np.linalg.det(R), 1., rtol=0, atol=1e-6)):
            raise FloatingPointError('Invalid comparison rotation')
        edges = np.diff(q, axis=0)
        lengths = np.linalg.norm(edges, axis=-1)
        length_error = float(np.max(np.abs(lengths-_numpy(model.rest))))
        rate_error = float(np.max(np.abs(np.sum(edges/lengths[:, None]*np.diff(v, axis=0), axis=-1))))
        if not np.isfinite(rate_error) or max(length_error, rate_error) > 1e-6:
            raise FloatingPointError('Cable constraint error exceeds 1e-6 m or 1e-6 m/s')
        max_length_error, max_rate_error = max(max_length_error, length_error), max(max_rate_error, rate_error)
        positions.append(np.array(p, copy=True))
        rotations.append(np.array(R, copy=True))
        nodes.append(q)
        velocities.append(v)
        if progress is not None:
            progress(i+1, total)
    check_cancelled()
    positions, rotations, nodes, velocities = map(np.asarray, (positions, rotations, nodes, velocities))
    markers = nodes[:, model.marker_indices[1:]]
    norm_rms = lambda error: float(np.sqrt(np.mean(np.sum(error**2, axis=-1))))
    angle = np.arccos(np.clip((np.einsum('nij,nij->n', rotations, replay.tracking_rotations[1:])-1)/2, -1, 1))
    metadata = deepcopy(replay.metadata)
    metadata.update({
        'schema': 'recorded_calibration_comparison_v1', 'mode': mode,
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'model_config': deepcopy(config), 'model_sha256': identities['model']['sha256'],
        'identities': identities, 'device': str(device) if cable is not None else 'cpu', 'dtype': 'float64',
        'physics_backend': 'cpp_tensor' if cable is not None and device != 'cpu' else 'native_cpu',
        'controller_period_s': None if cable is not None else runtime.controller_period_s,
        'controller_timing': ('not used' if cable is not None else
            'sampled geometric controller, phase zero at replay start; not verified firmware timing'),
        'motor_initialization': None if cable is not None else config['actuator']['initial_thrust_n'],
        'motor_initialization_source': 'configured provisional thrusts; not observed in old logs',
        'runtime_command_delay_s': None if cable is not None else 0.,
        'drone_motion_prescribed': cable is not None,
        'residual_enabled': model.residual is not None,
        'numerics': {'max_length_error_m': max_length_error, 'max_length_rate_error_mps': max_rate_error,
                     'length_tolerance_m': 1e-6, 'length_rate_tolerance_mps': 1e-6},
        'metrics': {
            'tracking_position_rmse_m': None if cable is not None else norm_rms(positions-replay.tracking_positions[1:]),
            'attitude_rmse_deg': None if cable is not None else float(np.rad2deg(np.sqrt(np.mean(angle**2)))),
            'cable_marker_rmse_m': norm_rms(markers-replay.cable_markers[1:]),
            'tip_rmse_m': norm_rms(markers[:, -1]-replay.cable_markers[1:, -1])}})
    arrays.update({
        'time_from_initial_s': np.arange(1, total+1)*model.dt,
        'measured_time_from_initial_s': replay.time_s[1:],
        'measured_tracking_positions': replay.tracking_positions[1:],
        'measured_tracking_rotations': replay.tracking_rotations[1:],
        'measured_cable_markers': replay.cable_markers[1:],
        'predicted_tracking_positions': positions, 'predicted_tracking_rotations': rotations,
        'predicted_cable_markers': markers, 'predicted_cable_nodes': nodes,
        'predicted_cable_velocities': velocities,
        'predicted_material_angles_rad': np.asarray(material_angles),
        'attachment_knot_time_s': replay.attachment_motion.time_s,
        'attachment_knot_positions': _numpy(replay.attachment_motion.positions),
        'attachment_knot_velocities': _numpy(replay.attachment_motion.velocities),
        'attachment_knot_rotations': _numpy(replay.attachment_motion.rotations),
        'attachment_tangent_body': _numpy(replay.attachment_motion.tangent_body),
        'controller_update_time_s': np.asarray(ticks, dtype=float)})
    if cable is not None:
        arrays['cable_on_support_force_world_n'] = np.asarray(reactions)
        arrays['cable_on_support_moment_world_nm'] = np.asarray(moments)
    else:
        arrays.update(native_command_arrival_time_s=replay.command_times_s, native_commands=replay.commands)
    verify_sources({'sources': metadata['sources']})
    for identity in identities.values():
        if digest(project_path(identity['path'])) != identity['sha256']:
            raise RuntimeError('Source code or model changed during comparison; result not accepted')
    return metadata, arrays


def save_comparison(metadata, arrays, output_root=None):
    """Save only a completed comparison, exclusively; never overwrite a run."""
    root = project_path(output_root if output_root is not None else ROOT/'runs/calibration')
    folder = root / datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    folder.mkdir(parents=True, exist_ok=False)
    path = folder/'comparison.json'
    archive = path.with_suffix('.npz')
    with archive.open('xb') as stream:
        np.savez_compressed(stream, **arrays)
    saved = deepcopy(metadata)
    saved['arrays_sha256'] = digest(archive)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(saved, stream, indent=2, allow_nan=False)
    return path
