"""Read-only presentation data. Never simulate, infer a policy, or score a rollout."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np


DEMO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = DEMO_ROOT / 'runs/ppo_stage4/20261007-034135-720530'
FINAL_SHA256 = 'f02b850b8f49cbeb06a0caf7883035eef991b6f1bf972c812bce738d7781c178'


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def velocity_metrics(velocity, normal):
    velocity = np.asarray(velocity, dtype=np.float64)
    normal = np.asarray(normal, dtype=np.float64)
    if (velocity.shape != (3,) or normal.shape != (3,)
            or not np.isfinite(velocity).all() or not np.isfinite(normal).all()
            or not np.isclose(np.linalg.norm(normal), 1, atol=1e-12, rtol=0)):
        raise ValueError('Finite XYZ velocity and unit target normal required')
    speed = float(np.linalg.norm(velocity))
    forward = float(velocity @ normal)
    direction = None if speed == 0 else velocity / speed
    angle = None if speed == 0 else float(np.degrees(np.arctan2(
        np.linalg.norm(np.cross(velocity, normal)), forward)))
    return dict(speed=speed, forward=forward, direction=direction, angle_deg=angle)


def strike_metrics(arrays, time_s, center, normal):
    """Same linear position/velocity interpolation as the saved strike scorer.

    This locates the saved selected crossing, not maximum speed or nearest point.
    No contact impulse or between-sample dynamics is reconstructed.
    """
    times = np.asarray(arrays['time_s'])
    if not np.isfinite(time_s) or not times[0] <= time_s <= times[-1]:
        raise ValueError('Selected strike time must lie within a saved interval')
    index = min(len(times) - 2, int(np.searchsorted(times, time_s, side='right') - 1))
    positions = arrays['cable_positions'][:, -1]
    velocities = arrays['cable_velocities'][:, -1]
    center, normal = np.asarray(center), np.asarray(normal)
    old = float((positions[index] - center) @ normal)
    new = float((positions[index + 1] - center) @ normal)
    if not old < 0 <= new:
        # A crossing exactly at an endpoint belongs to its preceding interval.
        if index and abs(time_s - times[index]) <= 1e-12:
            index -= 1
            old = float((positions[index] - center) @ normal)
            new = float((positions[index + 1] - center) @ normal)
        if not old < 0 <= new:
            raise ValueError('Recorded selected event is not a forward target-plane crossing')
    fraction = -old / (new - old)
    crossing_time = float(times[index] + fraction * (times[index + 1] - times[index]))
    if not np.isclose(crossing_time, time_s, atol=1e-9, rtol=0):
        raise ValueError('Saved scoring event time disagrees with the replay crossing')
    position = positions[index] + fraction * (positions[index + 1] - positions[index])
    velocity = velocities[index] + fraction * (velocities[index + 1] - velocities[index])
    side = np.array([-normal[1], normal[0], 0.])
    delta = position - center
    result = velocity_metrics(velocity, normal)
    result.update(time_s=crossing_time, position=position, velocity=velocity,
                  lateral_m=float(delta @ side), vertical_m=float(delta[2]),
                  center_error_m=float(np.hypot(delta @ side, delta[2])), interval=index)
    return result


def verify_recorded_metrics(metrics, event):
    components = event.get('components', {})
    fields = dict(speed='tip_speed_mps', forward='forward_mps',
                  lateral_m='target_lateral_error_m', vertical_m='target_vertical_error_m')
    for measured, saved in fields.items():
        if saved in components and not np.isclose(metrics[measured], components[saved], atol=1e-9, rtol=1e-9):
            raise ValueError('Presentation/replay mismatch: ' + saved)
    if 'direction_error_rad' in components and not np.isclose(
            np.pi if metrics['angle_deg'] is None else np.radians(metrics['angle_deg']),
            components['direction_error_rad'], atol=1e-9, rtol=1e-9):
        raise ValueError('Presentation/replay mismatch: direction_error_rad')


@dataclass
class PresentationData:
    directory: Path
    filename: str
    event: dict
    summary: dict
    snapshot: dict
    scene: object
    config: object
    replay: object
    surface: dict
    strike: dict
    hashes: dict

    @classmethod
    def load(cls, directory=DEFAULT_RUN, filename=None, event=None):
        from simulation.configuration import load_configuration
        from training.configuration import task_configuration
        from viewer.replay import StateReplay
        directory = Path(directory).resolve()
        rows = read_events(directory)
        if filename is None:
            evaluations = [row for row in rows if row.get('phase') == 'evaluation']
            if not evaluations:
                raise ValueError('No saved evaluation yet; no rollout will be started')
            event = evaluations[-1]
            filename = event.get('replay')
            if not filename:
                raise ValueError('Latest evaluation has no saved replay; no older result substituted')
        if Path(filename).name != filename:
            raise ValueError('Select an NPZ basename inside the run')
        if event is None:
            matching = [row for row in rows if row.get('replay') == filename]
            event = matching[-1] if matching else {}
        if event.get('replay') and event['replay'] != filename:
            raise ValueError('Event and selected replay reference disagree')
        summary_path = directory / 'summary.json'
        summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
        snapshot = json.loads((directory / 'configuration.json').read_text())
        scene = load_configuration(directory / 'scene.toml')
        case_path = directory / 'case.json'
        if event.get('protocol') == 'stationary_target_policy_test_v1':
            from evaluation.policy_test import case_scene
            from training.configuration import digest
            case = json.loads(case_path.read_text())
            request = json.loads((directory/'request.json').read_text())
            if event.get('case') != case or request.get('case_sha256') != digest(case):
                raise ValueError('Saved test case differs from its frozen request/result')
            if event.get('target_surface',{}).get('center_world_m') != case['target_position_world_m']:
                raise ValueError('Saved test surface differs from its actual case target')
            scene, _ = case_scene(scene,case)
        config = task_configuration(snapshot['task'], snapshot['settings'], snapshot['model'], scene)
        with np.load(directory / filename, allow_pickle=False) as data:
            replay = StateReplay(dict(data))
        surface = config.strike_surface.description(scene.initial_state.position_world_m, config.request)
        if 'target_surface' in event:
            selected = event['target_surface']
            center = np.asarray(selected['center_world_m'],dtype=np.float64)
            normal = np.asarray(selected['normal_world'],dtype=np.float64)
            if (center.shape != (3,) or normal.shape != (3,) or not np.isfinite(center).all()
                    or not np.isfinite(normal).all() or not np.allclose(normal,surface['normal_world'],rtol=0,atol=1e-12)
                    or any(selected[k] != surface[k] for k in ('width_m','height_m'))):
                raise ValueError('Invalid saved stationary target surface')
            surface = dict(selected)
        components = event.get('components', {})
        strike = None
        selected_time = components.get('release_time_s')
        if selected_time is not None:
            if components.get('plane_crossing') == 1:
                strike = strike_metrics(replay.arrays, selected_time,
                                        surface['center_world_m'], surface['normal_world'])
                verify_recorded_metrics(strike, event)
                strike['kind'] = 'Target crossing'
            else:
                strike = sampled_event_metrics(replay.arrays, selected_time,
                                               surface['center_world_m'], surface['normal_world'])
                # Stage1/2 reward selects a mechanics/proximity event, not a hit.
                verify_recorded_metrics(strike, event)
                strike['kind'] = 'Scored whip event (not a selected target crossing)'
        hashes = {name: sha256(directory / name) for name in
                  (filename, 'configuration.json', 'scene.toml', 'summary.json', 'latest.pt', 'case.json', 'request.json')
                  if (directory / name).exists()}
        if directory == DEFAULT_RUN.resolve() and hashes.get('latest.pt') != FINAL_SHA256:
            raise ValueError('Protected nominal FINAL checkpoint hash changed')
        return cls(directory, filename, event, summary, snapshot, scene, config, replay, surface, strike, hashes)

    @property
    def stage(self):
        return self.snapshot['settings'].get('training_stage', 1)

    def evidence(self):
        return dict(run=str(self.directory), replay=self.filename, stage=self.stage,
                    selected_round=self.event.get('round'), final_round=self.summary.get('round'),
                    collections=self.summary.get('round'), episodes=self.summary.get('episodes'),
                    saved_states_only=True, physics_steps=0, policy_inference_steps=0,
                    selected_event=None if self.strike is None else {
                        key: value.tolist() if isinstance(value, np.ndarray) else value
                        for key, value in self.strike.items()},
                    recorded_event=self.event, workspace=self.event.get('workspace', {}),
                    sha256=self.hashes,
                    limitations=['One selected saved motion, not robustness or real-flight evidence',
                                 'Virtual target plane: no target contact/impact force',
                                 'Linear interpolation of saved states, not unsaved physics substeps',
                                 'Selected mechanics event is not necessarily a crossing or a hit',
                                 'Latest checkpoint hash does not identify every historical replay actor',
                                 'Full saved paths include future samples as visual context, not policy inputs'])


def read_events(directory):
    """Ignore a writer's incomplete last line, not complete invalid evaluations."""
    source = Path(directory) / 'events.jsonl'
    if not source.exists():
        return []
    return [json.loads(line) for line in source.read_bytes().splitlines(keepends=True)
            if line.endswith(b'\n')]


def sampled_event_metrics(arrays, time_s, center, normal):
    times = arrays['time_s']
    if not np.isfinite(time_s) or not times[0] <= time_s <= times[-1]:
        raise ValueError('Selected event time lies outside the saved replay')
    position = np.array([np.interp(time_s, times, arrays['cable_positions'][:, -1, k])
                         for k in range(3)])
    velocity = np.array([np.interp(time_s, times, arrays['cable_velocities'][:, -1, k])
                         for k in range(3)])
    delta = position - np.asarray(center)
    side = np.array([-normal[1], normal[0], 0.])
    metrics = velocity_metrics(velocity, normal)
    metrics.update(time_s=float(time_s), position=position, velocity=velocity,
                   lateral_m=float(delta @ side), vertical_m=float(delta[2]),
                   center_error_m=float(np.hypot(delta @ side, delta[2])),
                   interval=int(np.searchsorted(times, time_s, side='right') - 1))
    return metrics
