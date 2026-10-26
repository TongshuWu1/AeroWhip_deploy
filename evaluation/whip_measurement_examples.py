"""Pin retained replays and explicit kinematic controls without running physics.

python -m evaluation.whip_measurement_examples --output <new-input-folder>
Synthetic arrays are geometric constructions, not dynamically feasible flights
or labels of whipping. Every case keeps its own model/task snapshot.
"""
import argparse
import json
from pathlib import Path
import tomllib

import numpy as np

from project_paths import ROOT, project_path
from viewer.replay_data import sha256


PRESET = 'configs/learning/whip_stage1_ppo_acceleration_discovery_v1.json'
RUN = 'runs/ppo_stage1/20261021-214537-277266'
RECHECK = 'tmp/whip-proxy-recheck-20261021'


def _chain(rest, angles, rates):
    """Integrate planar unit tangents; velocities are analytic derivatives."""
    edges = rest[None, :, None]*np.stack((np.sin(angles), np.zeros_like(angles),
                                        -np.cos(angles)), axis=-1)
    changes = rest[None, :, None]*rates[..., None]*np.stack(
        (np.cos(angles), np.zeros_like(angles), np.sin(angles)), axis=-1)
    zero = np.zeros((len(angles), 1, 3))
    return (np.concatenate((zero, np.cumsum(edges, axis=1)), axis=1),
            np.concatenate((zero, np.cumsum(changes, axis=1)), axis=1))


def kinematic_controls(rest_lengths, *, time_s=None, origin=(0., 0., 2.),
                       attachment_offset=(0., 0., -.051)):
    """Return deterministic, constant-edge-length controls and construction notes.

    Time and length scales below define fixtures, never classification thresholds.
    """
    rest = np.asarray(rest_lengths, dtype=float)
    t = np.arange(451)*.01 if time_s is None else np.asarray(time_s, dtype=float)
    if (rest.ndim != 1 or len(rest) < 2 or not np.isfinite(rest).all()
            or np.any(rest <= 0) or t.ndim != 1 or len(t) < 2
            or not np.isfinite(t).all() or np.any(np.diff(t) <= 0)):
        raise ValueError('Positive edge lengths and increasing finite times required')
    origin, offset = np.asarray(origin, float), np.asarray(attachment_offset, float)
    if (origin.shape != (3,) or offset.shape != (3,)
            or not np.isfinite(origin).all() or not np.isfinite(offset).all()):
        raise ValueError('Finite XYZ origin and attachment offset required')
    length = rest.sum()
    material = np.cumsum(rest)[:-1]
    time = t-t[0]
    duration = time[-1]

    def wrap(p, v, shift=None, shift_rate=None):
        shift = np.zeros((len(t), 3)) if shift is None else shift
        shift_rate = np.zeros_like(shift) if shift_rate is None else shift_rate
        return dict(time_s=t.copy(), cable_positions=p+origin+offset+shift[:, None],
                    cable_velocities=v+shift_rate[:, None],
                    drone_positions=origin+shift, drone_velocities=shift_rate.copy(),
                    drone_rotations=np.tile(np.eye(3), (len(t), 1, 1)),
                    drone_angular_velocities=np.zeros_like(shift),
                    state_valid=np.ones(len(t), bool),
                    cable_marker_indices=np.arange(len(rest)+1),
                    evidence_kind=np.asarray('synthetic_kinematic_control'))

    def wave(amplitude, backward=False, horizontal=False):
        # Prescribe signed turns at interior material points, yielding one
        # positive curvature packet instead of identifying a moving centroid.
        speed = 1.8*length/duration
        center = (-.25*length+speed*time if not backward else
                  1.25*length-speed*time)
        rate = -speed if backward else speed
        width = .12*length
        turns = amplitude*np.exp(-.5*((material[None]-center[:, None])/width)**2)
        turn_rate = turns*(material[None]-center[:, None])*rate/width**2
        angles = np.column_stack((np.zeros(len(t)), np.cumsum(turns, axis=1)))
        rates = np.column_stack((np.zeros(len(t)), np.cumsum(turn_rate, axis=1)))
        if horizontal:
            angles += np.pi/2
        return _chain(rest, angles, rates)

    base = np.tile(np.linspace(0., .9, len(rest)), (len(t), 1))
    p, _ = _chain(rest, base, np.zeros_like(base))
    theta = 2.*time
    cosine, sine = np.cos(theta), np.sin(theta)
    rotation = np.zeros((len(t), 3, 3))
    rotation[:, 0, 0] = rotation[:, 2, 2] = cosine
    rotation[:, 0, 2], rotation[:, 2, 0], rotation[:, 1, 1] = sine, -sine, 1.
    rotated = np.einsum('tij,tnj->tni', rotation, p)
    velocity = np.cross(np.array([0., 2., 0.]), rotated)
    result = [dict(name='rigid_curved_rotation', arrays=wrap(rotated, velocity),
                   notes='Fixed curved material shape rotates about the attachment at 2 rad/s; analytic velocities. No shape change, though the tip moves quickly.')]
    shift = np.column_stack((.8*time, np.zeros_like(time), np.zeros_like(time)))
    result.append(dict(name='translated_curved_shape', arrays=wrap(
        p, np.zeros_like(p), shift, np.tile([.8, 0., 0.], (len(t), 1))),
        notes='Unchanging curved cable translated uniformly at 0.8 m/s along world X.'))
    for name, backwards in [('small_travelling_wave', False), ('small_backward_wave', True)]:
        pos, vel = wave(.01, backward=backwards)
        result.append(dict(name=name, arrays=wrap(pos, vel), notes=(
            'Prescribed 0.01-rad Gaussian interior-turn packet moves '+
            ('tip toward attachment' if backwards else 'attachment toward tip')+
            ' in material coordinates; small amplitude is specified, not a whip label.')))
    fraction = time/duration
    turns = np.zeros((len(t), len(rest)-1))
    rates = np.zeros_like(turns)
    left, right = max(0, len(rest)//4-1), min(len(rest)-2, 3*len(rest)//4)
    turns[:, left], turns[:, right] = .25*(1-fraction), .25*fraction
    rates[:, left], rates[:, right] = -.25/duration, .25/duration
    angles = np.column_stack((np.zeros(len(t)), np.cumsum(turns, axis=1)))
    angular_rate = np.column_stack((np.zeros(len(t)), np.cumsum(rates, axis=1)))
    pos, vel = _chain(rest, angles, angular_rate)
    result.append(dict(name='separated_bends_fade_and_grow', arrays=wrap(pos, vel),
        notes='Two bends stay at distinct material nodes while one fades and the other grows; their curvature centroid shifts without a bend travelling between them.'))
    # A compactly supported time envelope makes the later segment exactly rigid.
    phase = np.clip(time/(.4*duration), 0., 1.)
    envelope = np.sin(np.pi*phase)**4
    envelope_rate = (4*np.pi/(.4*duration)*np.sin(np.pi*phase)**3*np.cos(np.pi*phase))
    envelope[phase >= 1.] = envelope_rate[phase >= 1.] = 0.
    center = -.2*length+3.5*length*time/duration
    width = .12*length
    gaussian = .3*np.exp(-.5*((material[None]-center[:, None])/width)**2)
    turns = gaussian*envelope[:, None]
    rates = gaussian*(envelope_rate[:, None]+envelope[:, None]*
                      (material[None]-center[:, None])*(3.5*length/duration)/width**2)
    angles = np.pi/2+np.column_stack((np.zeros(len(t)), np.cumsum(turns, axis=1)))
    angular_rate = np.column_stack((np.zeros(len(t)), np.cumsum(rates, axis=1)))
    pos, vel = _chain(rest, angles, angular_rate)
    phase = np.clip((time/duration-.72)/.13, 0., 1.)
    shift = np.zeros((len(t), 3)); shift_rate = np.zeros_like(shift)
    shift[:, 0] = .6*(3*phase**2-2*phase**3)
    shift_rate[:, 0] = .6*(6*phase-6*phase**2)/(.13*duration)
    arrays = wrap(pos, vel, shift, shift_rate)
    index = lambda f: int(np.argmin(abs(time-f*duration)))
    result.append(dict(name='earlier_wave_later_rigid_crossing', arrays=arrays,
        events=[dict(start_index=0, peak_index=index(.22), end_index=index(.4)),
                dict(start_index=index(.65), peak_index=index(.79), end_index=index(.95))],
        notes='Prescribed early travelling turn packet becomes exactly zero by 40% of duration. A later rigid horizontal translation crosses the frozen +X target plane. Explicit windows are construction intervals, not physical labels.'))
    return result


def generate(output):
    """Create a new bundle, checking retained provenance before writing anything."""
    from training.configuration import load
    output = project_path(output)
    if output.exists():
        raise FileExistsError(output)
    _, loaded, _, _ = load(PRESET)
    run_snapshot = ROOT/RUN/'configuration.json'
    frozen = json.loads(run_snapshot.read_text())
    if any(loaded[key] != frozen[key] for key in ('model', 'task', 'settings', 'scene_text')):
        raise ValueError('Current discovery metadata differs from its retained snapshot')
    previous = json.loads((ROOT/RECHECK/'report.json').read_text())
    controls = {row['name']: row for row in previous['native_controls']}
    sources = []
    for name, old in [('retained_hover', 'hover'), ('retained_small_sinusoid', 'cos_A1.0_f1.0'),
                      ('retained_strong_sinusoid', 'cos_A8.0_f1.0')]:
        record = controls[old]
        source = ROOT/record['path']
        if sha256(source) != record['sha256']:
            raise ValueError('Retained control hash mismatch: '+str(source))
        sources.append((name, source, run_snapshot,
            'Retained native control; no new simulation. Snapshot basis checked against the preset used by the retained control script.'))
    historical = ROOT/'models/milestone-20261009/evidence/stage1'
    sources.append(('historical_stage1_final', historical/'final-replay.npz',
                    historical/'configuration.json',
                    'Historical nominal Stage1 replay with its original snapshot. Different policy and action interface; not a fair training comparison or a ground-truth whip label.'))
    sources.append(('collection9_zero_noise_diagnostic', ROOT/'tmp/exploration-review-20261021/checkpoint9-seed2026102101-scale0.0-pairedlane0.npz',
                    run_snapshot, 'Retained separate zero-noise evaluation of the frozen round9 actor, nominal paired lane0. This is not a scheduled collection9 evaluation or an independent robustness trial.'))
    pinned = [(name, replay, snapshot, notes, sha256(replay), sha256(snapshot))
              for name, replay, snapshot, notes in sources]
    scene = tomllib.loads(loaded['scene_text'])
    synthetic = kinematic_controls(loaded['model']['cable']['marker_interval_lengths_m'],
        origin=scene['initial_state']['position_world_m'],
        attachment_offset=scene['platform']['attachment_offset_body_m'])
    output.mkdir(parents=True, exist_ok=False)
    cases = []
    for name, replay, snapshot, notes, digest, snapshot_digest in pinned:
        # Keep run/package replays attached to their original recorded outcome.
        # Detached numerical probes need a colocated frozen scene for playback.
        if (replay.parent == snapshot.parent and snapshot.name == 'configuration.json'
                and (snapshot.parent/'scene.toml').read_text(encoding='utf-8-sig') ==
                json.loads(snapshot.read_text())['scene_text']):
            snapshot_ref = str(snapshot)
        else:
            folder = output/name; folder.mkdir()
            copy = folder/'source_snapshot.json'
            copy.write_bytes(snapshot.read_bytes())
            (folder/'scene.toml').write_text(json.loads(copy.read_text())['scene_text'])
            snapshot_ref = str(copy.relative_to(output))
        cases.append(dict(name=name, path=str(replay), sha256=digest,
            source_snapshot=dict(path=snapshot_ref, sha256=snapshot_digest),
            original_snapshot=dict(path=str(snapshot), sha256=snapshot_digest),
            evidence_kind='saved_simulation', notes=notes))
    for control in synthetic:
        folder = output/control['name']; folder.mkdir()
        snapshot = folder/'source_snapshot.json'
        snapshot.write_text(json.dumps(loaded, indent=2, allow_nan=False)+'\n')
        (folder/'scene.toml').write_text(loaded['scene_text'])
        replay = folder/'replay.npz'
        np.savez_compressed(replay, **control['arrays'])
        case = dict(name=control['name'], path=str(replay.relative_to(output)), sha256=sha256(replay),
            source_snapshot=dict(path=str(snapshot.relative_to(output)), sha256=sha256(snapshot)),
            evidence_kind='synthetic_kinematic_control', notes=control['notes'])
        if 'events' in control:
            case['events'] = control['events']
        cases.append(case)
    manifest = dict(schema='offline_whip_event_inputs_v1', cases=cases,
        scope='Read-only retained histories plus prescribed kinematics; no physics, inference, training, reward ranking, or whip labels.',
        generator=dict(path=str(Path(__file__).resolve()), sha256=sha256(__file__)),
        provenance_reports=[dict(path=str(ROOT/p), sha256=sha256(ROOT/p)) for p in (
            RECHECK+'/report.json', 'tmp/exploration-review-20261021/report.json')])
    path = output/'manifest.json'
    path.write_text(json.dumps(manifest, indent=2, allow_nan=False)+'\n')
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    print(generate(parser.parse_args().output))


if __name__ == '__main__':
    main()
