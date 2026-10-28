"""Cable residual targets: per measured frame, the force on each cable node that the measured motion
requires and the model does not provide.

Measured: marker positions, velocities and accelerations from a Savitzky–Golay fit (cubic, --window
frames) over runs of consecutive camera frames with every marker seen; the clamp (attachment point)
moves with the drone pose, its velocity and acceleration from the same filter, the body rate from the
rotation. Model: `prescribed_acceleration` at the measured state, projected onto the cable lengths as
the replays do (positions along the measured edge directions at the rest lengths, velocities with the
stretching components removed); it returns the free-node accelerations from gravity, bending, damping
and air drag, before the inextensibility tensions. Target: F = m (a_measured − a_model) per node,
with the part a change of the edge tensions could produce removed (mass-weighted projection onto the
complement of the tension directions), so what remains is force the model has no way to make.

Saved per node-frame: the target force (N, world), the node's position and velocity relative to the
drone's tracking origin in the body frame, its world velocity, the local tangent, the arc position,
the drone's world velocity and its thrust direction and size (|a + g|, measured), plus the flight
name and role. Used by scripts/residual_feasibility.py.

    python scripts/residual_targets.py --output runs/residual/targets_<time>.npz
"""
import argparse
import math
from pathlib import Path
import sys
import time

import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aerowhip.fitting.calibration_set import folders, is_held_out
from aerowhip.fitting.least_squares import ProcessRunner
from aerowhip.physics.model import Model
from aerowhip.physics.model_configuration import DEFAULT_MODEL, load_model_configuration
from aerowhip.takes.calibration_flights import DT, load


def runs(ok, minimum):
    """(start, stop) of runs of True of at least `minimum` frames."""
    edges = np.flatnonzero(np.diff(np.r_[0, ok.astype(int), 0]))
    return [(a, b) for a, b in zip(edges[::2], edges[1::2]) if b-a >= minimum]


def project(root, root_v, markers, velocities, rest):
    """Positions at the rest lengths along the measured edge directions; velocities without stretching."""
    q, v = np.empty_like(markers), np.empty_like(velocities)
    previous, previous_v = root, root_v
    for i in range(len(rest)):
        edge = markers[i]-previous
        t = edge/np.linalg.norm(edge)
        q[i] = previous+t*rest[i]
        relative = velocities[i]-previous_v
        v[i] = previous_v+relative-t*(t@relative)
        previous, previous_v = q[i], v[i]
    return q, v


def tension_free(force, q, root, mass):
    """The part of the node forces (n, 3) no change of edge tensions can produce: mass-weighted
    projection onto the complement of the tension directions."""
    n = len(q)
    points = np.vstack([root, q])
    G = np.zeros((3*n, n))                    # column i: unit tension in edge i (root-node1 is edge 0)
    for i in range(n):
        t = points[i+1]-points[i]
        t /= np.linalg.norm(t)
        G[3*i:3*i+3, i] -= t                   # pulls node i+1 back toward node i
        if i > 0:
            G[3*(i-1):3*i, i] += t             # and node i toward node i+1
    w = np.repeat(1./mass, 3)
    F = force.ravel()
    lam = np.linalg.solve(G.T@(w[:, None]*G), G.T@(w*F))
    return (F-G@lam).reshape(n, 3)


def flight_targets(item):
    folder, config, window, role = item
    model = Model(config)
    native, info = model.native, model.native.info()
    rest = np.asarray(info['rest'])
    mass = np.asarray(info['node_mass'])[1:]
    attachment = np.asarray(info['attachment_tracking'])
    gravity = np.asarray(config['simulation']['gravity_world_m_s2'])
    f = load(folder)
    if f.markers is None:
        return None
    seen = np.isfinite(f.markers).all((1, 2))
    consecutive = np.r_[True, np.diff(f.frame) == 1]
    ok = seen & consecutive
    ok[1:] &= consecutive[1:]
    half = window//2
    out = {k: [] for k in ('force', 'rel_position', 'rel_velocity', 'world_velocity', 'tangent', 'arc', 'drone_velocity',
                           'thrust', 'frame', 'node', 'model_acceleration', 'measured_acceleration')}
    arc = np.cumsum(rest)
    R_all = Rotation.from_quat(f.quaternion)
    for a, b in runs(ok, window+2):
        sl = slice(a, b)
        P = f.position[sl]
        R = R_all[sl].as_matrix()
        root = P+R@attachment
        M = f.markers[sl]
        sg = lambda x, d: savgol_filter(x, window, 3, deriv=d, delta=DT, axis=0)
        root_v, root_a = sg(root, 1), sg(root, 2)
        Ps, Pv, Pa = sg(P, 0), sg(P, 1), sg(P, 2)
        Mv, Ma = sg(M, 1), sg(M, 2)
        Ms = sg(M, 0)
        rot = R_all[sl]
        for k in range(half, b-a-half):
            omega = (rot[k-1].inv()*rot[k+1]).as_rotvec()/(2*DT)
            q, v = project(root[k], root_v[k], Ms[k], Mv[k], rest)
            boundary = np.r_[root[k], root_v[k], root_a[k], R[k].ravel(), omega]
            try:
                acc = native.prescribed_acceleration(np.r_[q.ravel(), v.ravel()], boundary)
            except Exception:
                continue
            a_model = np.asarray(acc[:3*len(rest)]).reshape(len(rest), 3)
            force = tension_free(mass[:, None]*(Ma[k]-a_model), q, root[k], mass)
            Rt = R[k].T
            tangent = np.diff(np.vstack([root[k], q]), axis=0)
            tangent /= np.linalg.norm(tangent, axis=1)[:, None]
            out['force'].append(force)
            out['rel_position'].append((q-Ps[k])@Rt.T)
            out['rel_velocity'].append((v-Pv[k])@Rt.T)
            out['world_velocity'].append(v)
            out['tangent'].append(tangent)
            out['arc'].append(arc)
            out['drone_velocity'].append(np.repeat(Pv[k][None], len(rest), 0))
            thrust = Pa[k]-gravity
            out['thrust'].append(np.repeat(np.r_[thrust/np.linalg.norm(thrust), np.linalg.norm(thrust)][None], len(rest), 0))
            out['frame'].append(np.full(len(rest), a+k))
            out['node'].append(np.arange(len(rest)))
            out['model_acceleration'].append(a_model)
            out['measured_acceleration'].append(Ma[k])
    if not out['force']:
        return None
    result = {k: np.concatenate(v) if v[0].ndim else np.array(v) for k, v in out.items()}
    result['role'] = role
    result['flight'] = Path(folder).relative_to(Path(folder).parents[2]).as_posix()
    return result


def _prepare(_):
    return flight_targets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--window', type=int, default=11, help='Savitzky–Golay window, frames (odd)')
    args = parser.parse_args()
    t0 = time.time()
    config = load_model_configuration(DEFAULT_MODEL)
    items = [(folder, config, args.window, 'held_out' if is_held_out(folder) else 'fit') for folder in folders().values()]
    runner = ProcessRunner(items, _prepare, chunks_per_worker=1)
    results = [r for r, message in runner(None) if r is not None]
    runner.close()
    keys = [k for k in results[0] if k not in ('role', 'flight')]
    data = {k: np.concatenate([r[k] for r in results]) for k in keys}
    data['flight'] = np.concatenate([np.full(len(r['frame']), r['flight']) for r in results])
    data['role'] = np.concatenate([np.full(len(r['frame']), r['role']) for r in results])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, window=args.window, **data)
    n = len(data['frame'])
    print(f'{n} node-frames ({n//10} frames) from {len(results)} flights in {time.time()-t0:.0f} s -> {args.output}')
    F = data['force']
    for role in ('fit', 'held_out'):
        sel = data['role'] == role
        print(f'  {role}: {sel.sum()//10} frames, target force RMS {np.sqrt(np.mean(np.sum(F[sel]**2, 1)))*1e3:.2f} mN per node; '
              f'mean (world) {np.round(F[sel].mean(0)*1e3, 3).tolist()} mN')


if __name__ == '__main__':
    main()
