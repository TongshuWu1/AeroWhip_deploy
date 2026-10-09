"""Source-verified measured replay windows, separate from legacy fit sampling."""

from dataclasses import dataclass
from pathlib import Path
from project_paths import project_reference

import numpy as np
import torch

from fitting.prepare_window import initialize, rotation_vector
from simulation.tensor_model import PlantState
from simulation.state import quaternion_xyzw_to_rotation_matrix
from take_processing.library import session_name, source_files, verify_sources
from take_processing.paths import resolve_recording_path
from take_processing.take_data import load_take


class AttachmentMotion:
    """Offline C1 Hermite root motion; acceleration is an interpolant derivative.

    Knot velocities may use neighboring measured samples; positions inside an
    interval also use its future endpoint. This is recorded-boundary replay, not an online
    estimator. Acceleration is right-sided at interior knots and left-sided at
    the final knot. Integrators must split at ``time_s`` and use the left limit
    for their ending force stage (e.g. math.nextafter(end, start)).
    The prescribed tangent follows measured attitude by piecewise SO(3)
    geodesic interpolation. It is continuous, not an angular-rate measurement;
    the material-frame law uses its full rotation and piecewise body angular
    rate. This rate is derived from the attitude interpolant, not a gyro or
    observed cable twist. It is discontinuous at recorded attitude knots.
    """

    def __init__(self, time_s, positions, velocities, rotations, tangent_body, *, device="cpu"):
        self.time_s = np.array(time_s, dtype=np.float64, copy=True)
        p, v = np.asarray(positions, dtype=float), np.asarray(velocities, dtype=float)
        if (self.time_s.ndim != 1 or len(self.time_s) < 2
                or not np.isfinite(self.time_s).all() or self.time_s[0] != 0
                or not np.allclose(np.diff(self.time_s), .01, atol=1e-6, rtol=0)
                or p.shape != (len(self.time_s), 3) or v.shape != p.shape
                or not np.isfinite(p).all() or not np.isfinite(v).all()):
            raise ValueError("Attachment motion requires complete native-100-Hz finite XYZ knots")
        self.positions = torch.tensor(p, dtype=torch.float64, device=device)
        self.velocities = torch.tensor(v, dtype=torch.float64, device=device)
        R = np.asarray(rotations, dtype=float)
        tangent = np.asarray(tangent_body, dtype=float)
        if (R.shape != (len(self.time_s), 3, 3) or not np.isfinite(R).all()
                or not np.allclose(np.swapaxes(R, 1, 2)@R, np.eye(3), atol=1e-8, rtol=0)
                or not np.allclose(np.linalg.det(R), 1., atol=1e-8, rtol=0)
                or tangent.shape != (3,) or not np.isfinite(tangent).all()
                or abs(np.linalg.norm(tangent)-1) > 1e-10):
            raise ValueError('Point-clamp replay requires proper measured rotations and a unit body tangent')
        self.rotations = torch.tensor(R.copy(), dtype=torch.float64, device=device)
        self.tangent_body = torch.tensor(tangent.copy(), dtype=torch.float64, device=device)
        # Piecewise SO(3) geodesic interpolation (quaternion SLERP equivalent).
        # Precompute Rodrigues coefficients for R_i exp(u log(R_i^T R_i+1))*ta.
        # These are measured-boundary interpolation data, not a torsion state.
        rv = np.stack([rotation_vector(relative) for relative in np.swapaxes(R[:-1], 1, 2)@R[1:]])
        angles = np.linalg.norm(rv, axis=1)
        axes = np.divide(rv, angles[:, None], out=np.zeros_like(rv), where=angles[:, None]>0)
        self._angles = torch.tensor(angles, dtype=torch.float64, device=device)
        self._body_rate = torch.tensor(rv/np.diff(self.time_s)[:,None], dtype=torch.float64, device=device)
        skew = np.zeros((len(axes),3,3))
        skew[:,0,1],skew[:,0,2] = -axes[:,2],axes[:,1]
        skew[:,1,0],skew[:,1,2] = axes[:,2],-axes[:,0]
        skew[:,2,0],skew[:,2,1] = -axes[:,1],axes[:,0]
        self._axis_skew = torch.tensor(skew, dtype=torch.float64, device=device)

    def native_samples(self, start_s, duration_s, maximum_step):
        """Vectorized sampling of the same interpolant, not coarser integration."""
        from simulation.timing import integration_steps
        count, h = integration_steps(duration_s, maximum_step)
        starts = start_s+np.arange(count)*h
        ends = start_s+(np.arange(count)+1)*h
        ends[-1] = start_s+duration_s
        times = np.column_stack((starts, (starts+ends)/2, np.nextafter(ends, starts), ends))
        if (not np.isfinite(times).all() or np.min(times)<0 or np.max(times)>self.time_s[-1]):
            raise ValueError('Attachment motion cannot extrapolate outside measured knots')
        t=times.ravel();i=np.minimum(np.searchsorted(self.time_s,t,side='right')-1,len(self.time_s)-2)
        dt=torch.as_tensor(self.time_s[i+1]-self.time_s[i],dtype=torch.float64,device=self.positions.device)
        u=torch.as_tensor(t-self.time_s[i],dtype=dt.dtype,device=dt.device)/dt
        u,dt=u[:,None],dt[:,None]
        p0,p1=self.positions[i],self.positions[i+1];v0,v1=self.velocities[i],self.velocities[i+1]
        p=(2*u**3-3*u**2+1)*p0+(u**3-2*u**2+u)*dt*v0+(-2*u**3+3*u**2)*p1+(u**3-u**2)*dt*v1
        v=(6*u*u-6*u)*(p0-p1)/dt+(3*u*u-4*u+1)*v0+(3*u*u-2*u)*v1
        a=(12*u-6)*(p0-p1)/(dt*dt)+(6*u-4)*v0/dt+(6*u-2)*v1/dt
        angle=(u[:,0]*self._angles[i])[:,None,None];K=self._axis_skew[i]
        R=self.rotations[i]@(torch.eye(3,dtype=K.dtype,device=K.device)+torch.sin(angle)*K+2*torch.sin(angle/2)**2*(K@K))
        values=torch.cat((p,v,a,R.flatten(1),self._body_rate[i]),dim=1).detach().cpu().numpy()
        return np.column_stack((ends-starts,values.reshape(count,84)))

    def __call__(self, time_s):
        t = float(time_s)
        if not np.isfinite(t) or t < 0 or t > self.time_s[-1]:
            raise ValueError("Attachment motion cannot extrapolate outside measured knots")
        i = min(int(np.searchsorted(self.time_s, t, side="right"))-1, len(self.time_s)-2)
        h = self.time_s[i+1]-self.time_s[i]
        u = (t-self.time_s[i])/h
        p0, p1 = self.positions[i:i+2]
        v0, v1 = self.velocities[i:i+2]
        p = (2*u**3-3*u**2+1)*p0 + (u**3-2*u**2+u)*h*v0
        p = p + (-2*u**3+3*u**2)*p1 + (u**3-u**2)*h*v1
        v = (6*u*u-6*u)*(p0-p1)/h + (3*u*u-4*u+1)*v0 + (3*u*u-2*u)*v1
        a = (12*u-6)*(p0-p1)/(h*h) + (6*u-4)*v0/h + (6*u-2)*v1/h
        angle = u*self._angles[i]
        K = self._axis_skew[i]
        rotation = self.rotations[i]@(torch.eye(3,dtype=K.dtype,device=K.device)
            +torch.sin(angle)*K+2*torch.sin(angle/2)**2*(K@K))
        return p, v, a, rotation, self._body_rate[i]


@dataclass(frozen=True)
class RecordedReplay:
    initial: PlantState
    time_s: np.ndarray
    tracking_positions: np.ndarray
    tracking_rotations: np.ndarray
    cable_markers: np.ndarray
    command_times_s: np.ndarray | None
    commands: np.ndarray | None
    attachment_motion: AttachmentMotion
    metadata: dict


def _command_events(rows, *, offset_s, source_start_s, duration_s, delay_s):
    """Convert receipt events to already-delayed model arrivals exactly once."""
    fields = [f"cmd_{v}" for v in ("x", "y", "z", "vx", "vy", "vz",
                                   "ax", "ay", "az", "yaw", "yaw_rate")]
    required = {"time_s", "cmd_valid", "cmd_sequence", *fields}
    if not required.issubset(rows.dtype.names or ()):
        raise ValueError("Command log lacks complete timestamped PVA/yaw/valid/sequence fields")
    if (not np.isfinite([offset_s, source_start_s, duration_s, delay_s]).all()
            or delay_s < 0 or duration_s <= 0):
        raise ValueError("Finite alignment/start and nonnegative delay/positive duration required")
    receipt = np.asarray(rows["time_s"], dtype=float)
    if len(receipt) < 2 or not np.isfinite(receipt).all() or np.any(np.diff(receipt) <= 0):
        raise ValueError("Command timestamps must be finite and strictly increasing")
    # receipt = tracking source + alignment offset; model delay is then added.
    arrivals = receipt-offset_s+delay_s-source_start_s
    held = int(np.searchsorted(arrivals, 0., side="right"))-1
    coverage = int(np.searchsorted(arrivals, duration_s, side="left"))
    if held < 0 or coverage >= len(arrivals):
        raise ValueError("Command log does not cover the complete delayed replay window")
    span = slice(held, coverage+1)
    values = np.column_stack([rows[field] for field in fields])
    sequence = np.asarray(rows["cmd_sequence"][span], dtype=float)
    if (not np.isfinite(values[span]).all() or np.any(rows["cmd_valid"][span] != 1)
            or not np.isfinite(sequence).all() or np.any(sequence != np.floor(sequence))
            or np.any(np.diff(sequence) != 1) or np.any(np.diff(receipt[span]) > .2)
            or -arrivals[held] > .2):
        raise ValueError("Invalid, missing or stale command events in selected replay window")
    rate_fields = [name for name in rows.dtype.names
                   if name in {"cmd_yaw_rate", "cmd_omega_x", "cmd_omega_y", "cmd_omega_z",
                               "cmd_roll_rate", "cmd_pitch_rate"}]
    for name in rate_fields:
        rate = rows[name][span]
        if not np.isfinite(rate).all() or np.any(rate != 0):
            raise ValueError(f"Unsupported body/yaw-rate feedforward: {name}")
    quaternion_fields = ["cmd_qx", "cmd_qy", "cmd_qz", "cmd_qw"]
    present = set(quaternion_fields).intersection(rows.dtype.names)
    if present:
        if len(present) != 4:
            raise ValueError("Command quaternion must contain all four XYZW fields")
        q = np.column_stack([rows[name][span] for name in quaternion_fields])
        if (not np.isfinite(q).all()
                or not np.allclose(np.linalg.norm(q, axis=1), 1., rtol=0, atol=1e-6)):
            raise ValueError("Command quaternions must be finite and normalized")
        if np.any(np.abs(q[:, :2]) > 1e-8):
            raise ValueError("Commanded roll/pitch attitude is unsupported by the PVA/yaw replay")
        quaternion_yaw = 2*np.arctan2(q[:, 2], q[:, 3])
        yaw_error = quaternion_yaw-values[span, 9]
        if np.any(np.abs(np.arctan2(np.sin(yaw_error), np.cos(yaw_error))) > 1e-6):
            raise ValueError("Command quaternion yaw disagrees with cmd_yaw")
    stop = int(np.searchsorted(arrivals, duration_s, side="right"))
    ids = np.arange(held, stop)
    times = arrivals[ids].copy()
    times[0] = 0.  # The command already held at initialization, not a new arrival.
    return times, values[ids].copy(), {
        "held_command_sequence": int(sequence[0]),
        "held_command_arrival_relative_s": float(arrivals[held]),
        "held_command_age_s": float(-arrivals[held]),
        "event_count": len(times), "coverage_arrival_relative_s": float(arrivals[coverage]),
        "command_delay_s": float(delay_s), "delay_already_applied": True,
        "command_sampling": "native receipt events mapped to model arrivals; zero-order hold",
        "maximum_allowed_command_gap_s": .2,
    }


def prepare_recorded_replay(path, model, *, start_s, duration_s, history_s=.1,
                            require_commands=True):
    """Select observed samples including t=0; never resample measurements.

    ``start_s`` selects the first native sample at/after that elapsed time, as
    in prepare_window. Commands, when requested, are already delayed and must
    bypass any further runtime command delay. ``require_commands=False`` is
    measured cable-boundary preparation and does not require saved alignment.
    """
    if not np.isfinite([start_s, duration_s, history_s]).all() or min(duration_s, history_s) <= 0:
        raise ValueError("Finite start and positive duration/history are required")
    dt = .01
    if not np.isfinite(model.dt) or abs(model.dt-dt) > 1e-12:
        raise ValueError("Recorded replay requires the model's native 10-ms output timestep")
    steps, history_steps = round(duration_s/dt), round(history_s/dt)
    if (steps < 1 or history_steps < 2 or abs(steps*dt-duration_s) > 1e-9
            or abs(history_steps*dt-history_s) > 1e-9):
        raise ValueError("Replay duration/history must be complete native 10-ms steps; history >=20 ms")
    path = resolve_recording_path(path)
    sources = source_files(path)
    take = load_take(path)
    start = int(np.searchsorted(take.time, start_s))
    first, last = start-history_steps, start+steps
    if start_s < 0 or first < 0 or last >= len(take.time):
        raise ValueError("Replay window or causal history lies outside the take")
    selected = slice(first, last+1)
    if (not np.allclose(np.diff(take.time[selected]), dt, atol=1e-6, rtol=0)
            or not np.allclose(np.diff(take.source_time[selected]), dt, atol=1e-6, rtol=0)
            or np.any(np.diff(take.frame[selected]) != 1)):
        raise ValueError("Replay/history contains missing frames or is not native 100-Hz data")
    positions = take.position[selected]
    markers = take.cable[selected]
    quaternions = take.quaternion[selected]
    if not all(np.isfinite(x).all() for x in (positions, markers, quaternions)):
        raise ValueError("Incomplete measured pose/cable replay or initialization history")
    if markers.shape != (len(positions), len(model.marker_indices)-1, 3):
        raise ValueError("Measured marker count does not match the model")
    rotations = np.stack([quaternion_xyzw_to_rotation_matrix(q) for q in quaternions])
    history = slice(0, history_steps+1)
    initial, initialization = initialize(model, take.time[first:start+1], positions[history],
                                         rotations[history], markers[history])
    time = take.time[start:last+1].copy()-take.time[start]
    # Individual intervals may be close to 10 ms while accumulating drift.
    # Coupled runtime and saved overlays use nominal output times, so reject
    # such windows instead of silently relabeling or resampling measurements.
    if not np.allclose(time, np.arange(steps+1)*dt, atol=1e-6, rtol=0):
        raise ValueError('Measured output clock drifts from the native 100-Hz replay grid')
    output = slice(history_steps, None)
    attachment = model.attachment_tracking.detach().cpu().numpy()
    roots = positions + np.einsum("nij,j->ni", rotations, attachment)
    # Offline three-point quadratic derivatives: centered at interior knots,
    # backward at the final knot. Include the existing history so a one-step
    # replay still has enough samples; never request data beyond its endpoint.
    velocities = np.gradient(roots, take.time[selected], axis=0, edge_order=2)[output].copy()
    # Exactly match initialize's rigid-body root velocity at the initial knot.
    root_velocity = initial.velocity_com + initial.rotation @ torch.linalg.cross(initial.omega_body, model.arm)
    velocities[0] = root_velocity.detach().cpu().numpy()
    boundary = AttachmentMotion(time, roots[output], velocities, rotations[output],
                                model.root_tangent_body.detach().cpu().numpy(), device=model.device)
    event_times = commands = None
    command_metadata = {"command_sampling": "not requested; measured attachment boundary only",
                        "delay_already_applied": False}
    if require_commands:
        if not take.alignment_applied:
            raise ValueError("A source-verified saved clock alignment is required for command replay")
        command_path = Path(sources["commands"]["path"])
        if not command_path.is_file():
            raise ValueError("A native command event log is required for command replay")
        rows = np.atleast_1d(np.genfromtxt(command_path, delimiter=",", names=True, encoding="utf-8-sig"))
        event_times, commands, command_metadata = _command_events(
            rows, offset_s=float(take.alignment["offset_s"]),
            source_start_s=float(take.source_time[start]), duration_s=float(time[-1]),
            delay_s=float(model.config["controller"]["command_delay_s"]))
    group, grouping = session_name(path)
    verify_sources({"sources": sources})
    sources = {key: dict(value, path=project_reference(value['path'])) for key, value in sources.items()}
    metadata = {
        "take": project_reference(path), "sources": sources, "recording_group": group, "grouping": grouping,
        "requested_start_elapsed_s": float(start_s), "start_elapsed_s": float(take.time[start]),
        "end_elapsed_s": float(take.time[last]), "start_source_s": float(take.source_time[start]),
        "end_source_s": float(take.source_time[last]), "start_frame": int(take.frame[start]),
        "end_frame": int(take.frame[last]), "alignment": take.alignment,
        "alignment_applied": bool(take.alignment_applied), "initialization": initialization,
        "measured_sampling": "native 100 Hz including initial sample; no resampling",
        "attachment_interpolation": "offline C1 cubic Hermite through measured p_tracking + R*a",
        "attachment_orientation_interpolation": "piecewise SO(3) geodesic interpolation through measured rotations",
        "attachment_angular_rate": "body rate of attitude interpolant; piecewise constant, not measured gyro or cable twist",
        "attachment_velocity_estimator": "offline three-point quadratic derivative (centered interior, backward final); initial knot uses causal rigid-body velocity",
        "attachment_velocity_stencil_samples": 3,
        "attachment_initial_velocity_history_s": float(history_s),
        "attachment_acceleration": "piecewise interpolant derivative, not measured acceleration; split integration at knots",
        "purpose": "recorded diagnostic replay; no fitting or model promotion",
        **command_metadata,
    }
    return RecordedReplay(initial, time, positions[output].copy(), rotations[output].copy(),
                          markers[output].copy(), event_times, commands, boundary, metadata)
