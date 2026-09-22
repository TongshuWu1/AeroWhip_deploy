"""Causal native-100-Hz flight windows for the coupled model."""
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
from fitting.coupled_fit import FitWindow
from take_processing.take_data import load_take
from take_processing.library import source_files, session_name, verify_sources
from simulation.state import quaternion_xyzw_to_rotation_matrix
from simulation.dynamics.coupled import CoupledState


def rotation_vector(R):
    angle = np.arccos(np.clip((np.trace(R)-1)/2, -1, 1))
    vector = np.array([R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]])
    if angle > np.pi-.01:
        raise ValueError("Initialization rotations span an ambiguous half turn")
    return vector*.5 if angle < 1e-7 else vector*angle/(2*np.sin(angle))


def initialize(model, time, positions, rotations, markers):
    """Pose at last history sample; velocities use only this causal history.

    Cable directions are measured, lengths configured; subdivisions are straight.
    Velocity correction enforces fixed-length constraints without moving the drone.
    """
    rest = model.rest.cpu().numpy()
    attachment = model.attachment_tracking.cpu().numpy()
    counts = np.diff(model.marker_indices)
    projected = []
    for p, R, measured in zip(positions, rotations, markers):
        root = p+R@attachment
        sites = np.vstack([root, measured])
        directions = np.diff(sites, axis=0)
        norms = np.linalg.norm(directions, axis=1)
        if np.any(norms < 1e-6):
            raise ValueError("Coincident cable markers in initialization history")
        directions /= norms[:, None]
        edges = np.repeat(directions, counts, axis=0)*rest[:, None]
        projected.append(np.vstack([root, root+np.cumsum(edges, axis=0)]))
    projected = np.asarray(projected)
    centered = time-time.mean()
    def slope(values):
        return np.tensordot(centered, values, axes=(0,0))/np.dot(centered, centered)
    R = rotations[-1]
    omega = slope(np.stack([rotation_vector(R.T@previous) for previous in rotations]))
    tracking_v = slope(positions)
    com = model.com_tracking.cpu().numpy()
    com_v = tracking_v+R@np.cross(omega, com)
    q = projected[-1]
    velocity = slope(projected)
    velocity[0] = tracking_v+R@np.cross(omega, attachment)
    correction_before = velocity.copy()
    for i, edge in enumerate(np.diff(q, axis=0)):
        tangent = edge/np.linalg.norm(edge)
        velocity[i+1] -= tangent*np.dot(tangent, velocity[i+1]-velocity[i])
    state = CoupledState(*(model.tensor(v) for v in (
        positions[-1]+R@com, R, com_v, omega, q[1:], velocity[1:])))
    adjustment = q[model.marker_indices[1:]]-markers[-1]
    return state, {
        "initial_marker_projection_rms_m": float(np.sqrt(np.mean(np.sum(adjustment**2,axis=1)))),
        "initial_marker_projection_max_m": float(np.linalg.norm(adjustment,axis=1).max()),
        "initial_velocity_constraint_correction_max_mps": float(np.linalg.norm(velocity-correction_before,axis=1).max()),
        "history_samples": len(time), "history_duration_s": float(time[-1]-time[0]),
    }


def prepare_window(path, model, *, start_s, duration_s=2.5, history_s=.1):
    path = Path(path).resolve()
    sources = source_files(path)
    entry = {"sources": sources}
    take = load_take(path)
    if not take.alignment_applied:
        raise ValueError("A source-verified saved clock alignment is required")
    if not np.isfinite([start_s, duration_s, history_s]).all() or min(duration_s, history_s) <= 0:
        raise ValueError("Finite start and positive duration/history are required")
    start = int(np.searchsorted(take.time, start_s))
    steps = round(duration_s/model.dt)
    history_steps = round(history_s/model.dt)
    if (steps < 1 or history_steps < 2 or start < history_steps or start+steps >= len(take.time)
            or abs(steps*model.dt-duration_s)>1e-9):
        raise ValueError("Window/history lies outside take or duration is not an integer model step")
    first, last = start-history_steps, start+steps
    selected = slice(first,last+1)
    if (not np.allclose(np.diff(take.time[selected]),model.dt,atol=1e-6,rtol=0)
            or np.any(np.diff(take.frame[selected])!=1)):
        raise ValueError("Window contains gaps or is not native model-rate data; no resampling is performed")
    for values in (take.position[selected],take.quaternion[selected],take.cable[selected]):
        if not np.isfinite(values).all():
            raise ValueError("Incomplete measured pose/cable window; select a different interval")
    rotations = np.stack([quaternion_xyzw_to_rotation_matrix(q) for q in take.quaternion[selected]])
    history = slice(first,start+1)
    state, initialization = initialize(model, take.time[history], take.position[history],
                                      rotations[:history_steps+1],take.cable[history])
    commands_path = Path(sources["commands"]["path"])
    commands = np.atleast_1d(np.genfromtxt(commands_path,delimiter=",",names=True,encoding="utf-8-sig"))
    fields = [f"cmd_{v}" for v in ("x","y","z","vx","vy","vz","ax","ay","az","yaw","yaw_rate")]
    values = np.column_stack([commands[field] for field in fields])
    times = commands['time_s']
    if not np.isfinite(times).all() or np.any(np.diff(times)<=0):
        raise ValueError("Invalid command timestamps")
    delay = float(model.config['controller']['command_delay_s'])
    if not np.isfinite(delay) or delay<0:
        raise ValueError("Command delay must be finite and nonnegative")
    # Receipt clock = Motive source clock + offset. Delay is applied once.
    queries = take.source_time[start:start+steps]+float(take.alignment['offset_s'])-delay
    ids = np.searchsorted(times,queries,side='right')-1
    if np.any(ids<0) or queries[-1]+model.dt>times[-1]:
        raise ValueError("Command log does not cover delayed window")
    span = slice(ids[0], min(len(times),np.searchsorted(times,queries[-1]+model.dt,side='right')+1))
    if (not np.isfinite(values[span]).all() or np.any(commands['cmd_valid'][span]!=1)
            or np.any(np.diff(commands['cmd_sequence'][span])!=1)
            or np.any(np.diff(times[span])>.2) or np.any(queries-times[ids]>.2)):
        raise ValueError("Invalid, missing or stale command events in selected window")
    group, grouping = session_name(path)
    if grouping=='folder':
        raise ValueError("Explicit recording-group metadata is required before fitting preparation")
    window = FitWindow(group,state,values[ids].copy(),take.position[start+1:last+1].copy(),
                       rotations[history_steps+1:].copy(),take.cable[start+1:last+1].copy(),model.dt)
    window.validate(model)
    verify_sources(entry)
    metadata = {"take":str(path),"recording_group":group,"sources":sources,
        "start_elapsed_s":float(take.time[start]),"end_elapsed_s":float(take.time[last]),
        "start_source_s":float(take.source_time[start]),"start_frame":int(take.frame[start]),
        "end_frame":int(take.frame[last]),"alignment":take.alignment,
        "command_delay_s":delay,"command_sampling":"left-endpoint zero-order hold at 100 Hz; event quantization < one step",
        "initialization":initialization,"purpose":"baseline diagnostic; no train/validation role assigned"}
    return window, metadata
