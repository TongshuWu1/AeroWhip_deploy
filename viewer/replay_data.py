"""Read-only presentation data. Never simulate, infer a policy, or score a rollout."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
from project_paths import project_path, contained_child


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
    hashes: dict
    measurement_event: dict | None = None

    @classmethod
    def load_measurement(cls, report_path, case_index=0, event_index=None):
        """Display a pinned offline review without rescoring or changing saved outcomes."""
        from simulation.configuration import load_configuration
        from training.configuration import task_configuration
        from viewer.replay import StateReplay
        report_path = project_path(report_path)
        report = json.loads(report_path.read_text())
        if report.get('schema') != 'offline_whip_event_review_v1':
            raise ValueError('Unsupported offline measurement report schema')
        cases = report.get('cases')
        if (not isinstance(cases, list) or type(case_index) is not int
                or not 0 <= case_index < len(cases)):
            raise ValueError('Measurement case index is out of range (indices start at zero)')
        case = cases[case_index]
        pinned = {}
        for field, digest in (('source_replay', 'replay_sha256'),
                              ('source_snapshot', 'source_snapshot_sha256')):
            value = case.get(field)
            if not isinstance(value, str) or not Path(value).is_absolute():
                raise ValueError('Measurement source must be an absolute path: '+field)
            path = project_path(value)
            if sha256(path) != case.get(digest):
                raise ValueError('Measurement source SHA256 mismatch: '+field)
            pinned[field] = path
        replay_path, snapshot_path = pinned['source_replay'], pinned['source_snapshot']
        snapshot = json.loads(snapshot_path.read_text())
        scene_path = contained_child(snapshot_path.parent, 'scene.toml')
        if scene_path.read_text(encoding='utf-8-sig') != snapshot.get('scene_text'):
            raise ValueError('Saved scene differs from the pinned configuration snapshot')
        # Ordinary training/package replays retain their exact recorded outcome.
        if replay_path.parent == snapshot_path.parent and snapshot_path.name == 'configuration.json':
            data = cls.load(replay_path.parent, replay_path.name)
        else:
            scene = load_configuration(scene_path)
            config = task_configuration(snapshot['task'], snapshot['settings'], snapshot['model'], scene)
            with np.load(replay_path, allow_pickle=False) as archive:
                replay = StateReplay(dict(archive))
            # Detached diagnostics have no recorded training outcome to infer.
            data = cls(replay_path.parent, replay_path.name, {}, {}, snapshot, scene,
                       config, replay, {})
        events = case.get('events')
        if not isinstance(events, list):
            raise ValueError('Measurement case must contain an event list')
        if event_index is None:
            event_index = case.get('default_event_index', 0 if events else None)
        candidate = None
        if event_index is not None:
            if type(event_index) is not int or not 0 <= event_index < len(events):
                raise ValueError('Measurement event index is out of range (indices start at zero)')
            candidate = events[event_index]
            if not isinstance(candidate.get('id'), (str, int)):
                raise ValueError('Measurement candidate must have an event id')
            indices = [candidate.get(key+'_index') for key in ('start', 'peak', 'end')]
            if (any(type(index) is not int for index in indices)
                    or not 0 <= indices[0] <= indices[1] <= indices[2] < len(data.replay.time)):
                raise ValueError('Measurement event indices do not describe a saved interval')
            for key, index in zip(('start', 'peak', 'end'), indices):
                value = candidate.get(key+'_s')
                if (not isinstance(value, (int, float)) or not np.isfinite(value)
                        or not np.isclose(value, data.replay.time[index], atol=1e-9, rtol=0)):
                    raise ValueError('Measurement event time/index mismatch: '+key)
            if candidate.get('review_status') != 'unreviewed':
                raise ValueError('This viewer accepts unreviewed measurement candidates only')
            notes = candidate.get('notes')
            if not isinstance(notes, list) or any(not isinstance(note, str) for note in notes):
                raise ValueError('Measurement candidate notes must be a list of strings')
        data.measurement_event = dict(report_path=str(report_path), case_index=case_index,
            case_name=case.get('name', str(case_index)), evidence_kind=case.get('evidence_kind', 'unspecified'),
            event_index=event_index, event=candidate, source_hashes=report.get('source_hashes', {}))
        data.hashes.update(measurement_report=sha256(report_path),
                           source_replay=case['replay_sha256'], source_snapshot=case['source_snapshot_sha256'])
        return data

    @classmethod
    def load(cls, directory, filename=None, event=None):
        from simulation.configuration import load_configuration
        from training.configuration import task_configuration
        from viewer.replay import StateReplay
        directory = project_path(directory)
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
        summary_path = contained_child(directory, 'summary.json')
        summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
        snapshot = json.loads(contained_child(directory, 'configuration.json').read_text())
        scene = load_configuration(directory / 'scene.toml')
        case_path = contained_child(directory, 'case.json')
        if event.get('protocol') == 'initial_state_policy_test_v1':
            from evaluation.policy_test import case_scene
            from training.configuration import digest
            case = json.loads(case_path.read_text())
            request = json.loads(contained_child(directory, 'request.json').read_text())
            if event.get('case') != case or request.get('case_sha256') != digest(case):
                raise ValueError('Saved test case differs from its frozen request/result')
            scene = case_scene(scene,case)
        config = task_configuration(snapshot['task'], snapshot['settings'], snapshot['model'], scene)
        with np.load(contained_child(directory, filename), allow_pickle=False) as data:
            replay = StateReplay(dict(data))
        hashes = {name: sha256(contained_child(directory, name)) for name in
                  (filename, 'configuration.json', 'scene.toml', 'summary.json', 'latest.pt', 'case.json', 'request.json')
                  if (directory / name).exists()}
        return cls(directory, filename, event, summary, snapshot, scene, config, replay, hashes)

    @property
    def output_direction(self):
        """Task output direction in world axes, for live tip-velocity display."""
        return self.config.output_direction_world

    def evidence(self):
        return dict(run=str(self.directory), replay=self.filename,
                    selected_round=self.event.get('round'), final_round=self.summary.get('round'),
                    collections=self.summary.get('round'), episodes=self.summary.get('episodes'),
                    saved_states_only=True, physics_steps=0, policy_inference_steps=0,
                    recorded_event=self.event, workspace=self.event.get('workspace', {}),
                    measurement_review=self.measurement_event,
                    sha256=self.hashes,
                    limitations=['One selected saved motion, not robustness or real-flight evidence',
                                 'Linear interpolation of saved states, not unsaved physics substeps',
                                 'Latest checkpoint hash does not identify every historical replay actor',
                                 'Full saved paths include future samples as visual context, not policy inputs'])


def read_events(directory):
    """Ignore a writer's incomplete last line, not complete invalid evaluations."""
    source = contained_child(directory, 'events.jsonl')
    if not source.exists():
        return []
    return [json.loads(line) for line in source.read_bytes().splitlines(keepends=True)
            if line.endswith(b'\n')]
