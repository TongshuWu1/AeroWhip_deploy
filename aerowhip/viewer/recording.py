"""Versioned display recording and indexed playback; no physics or ROS runtime."""
from dataclasses import asdict, is_dataclass, replace
from datetime import datetime, timezone
from enum import Enum
import json
import math
import os
from pathlib import Path
from time import perf_counter
import uuid

import numpy as np

from aerowhip.physics.configuration import InitialStateConfiguration, PlatformConfiguration, SimulationConfiguration, ViewerConfiguration
from aerowhip.physics.state import DroneState, ForceTelemetry
from aerowhip.viewer.display_state import CableDisplay, CableTrackingDisplay, DisplayFrame, RobotDisplay
from aerowhip.viewer.sources import ReplaySource, SourceSnapshot

SCHEMA = 'aerowhip_display_recording_v1'


def json_value(value):
    if is_dataclass(value):
        return json_value(asdict(value))
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {key: json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(item) for item in value]
    return value


def decode_frame(data):
    robots = []
    for item in data['robots']:
        robot = dict(item)
        robot['drone'] = DroneState(**robot['drone'])
        if robot.get('cable') is not None:
            cable = dict(robot['cable'])
            cable['marker_indices'] = np.asarray(cable['marker_indices'], dtype=np.int64)
            robot['cable'] = CableDisplay(**cable)
        if robot.get('cable_tracking') is not None:
            tracking = dict(robot['cable_tracking'])
            tracking['raw_points_world_m'] = np.asarray(tracking['raw_points_world_m']).reshape(-1, 3)
            robot['cable_tracking'] = CableTrackingDisplay(**tracking)
        if robot.get('forces') is not None:
            robot['forces'] = ForceTelemetry(**robot['forces'])
        robots.append(RobotDisplay(**robot))
    return DisplayFrame(**dict(data, robots=tuple(robots)))


def decode_configuration(data):
    def arrays(values):
        return {key: np.asarray(value, dtype=float) if isinstance(value, list) else value
                for key, value in values.items()}
    return SimulationConfiguration(
        initial_state=InitialStateConfiguration(**arrays(data['initial_state'])),
        platform=PlatformConfiguration(**arrays(data['platform'])),
        viewer=ViewerConfiguration(**data['viewer']), timestep_s=data['timestep_s'],
        physics_backend=data['physics_backend'])


class DisplayRecorder:
    """Called under the feed's owner lock, never from the render timer.

    Line buffering retains complete prefixes after interruption. Stop flushes and
    fsyncs both streams before atomically marking the manifest complete.
    """
    def __init__(self, parent, configuration, metadata, snapshot):
        self.directory = Path(parent)/f'{datetime.now(timezone.utc):%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:8]}'
        self.directory.mkdir(parents=True, exist_ok=False)
        self.manifest = self.directory/'recording.json'
        self.started = perf_counter()
        self.frames = self.events = None
        self.error = None
        self.frame_count = self.event_count = 0
        self.last_elapsed = 0.
        self.active = True
        self.metadata = dict(schema=SCHEMA, complete=False,
            created_utc=datetime.now(timezone.utc).isoformat(),
            configuration=json_value(configuration), session=json_value(metadata),
            frames='frames.jsonl', events='events.jsonl',
            timeline='Elapsed host monotonic recording time; original source timestamps preserved in frames/events',
            meaning='Saved display results and received samples; not a restartable physics checkpoint or lossless camera recording')
        try:
            self._manifest()
            self.frames = (self.directory/'frames.jsonl').open('x', encoding='utf-8', buffering=1)
            self.events = (self.directory/'events.jsonl').open('x', encoding='utf-8', buffering=1)
            self.frame(snapshot, initial=True)
        except Exception:
            for handle in (self.frames, self.events):
                if handle is not None:
                    handle.close()
            raise

    def _manifest(self):
        temporary = self.directory/'recording.json.tmp'
        with temporary.open('w', encoding='utf-8') as handle:
            json.dump(self.metadata, handle, indent=2, allow_nan=False)
            handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, self.manifest)

    def _write(self, handle, value):
        if not self.active:
            return False
        try:
            handle.write(json.dumps(json_value(value), allow_nan=False, separators=(',', ':'))+'\n')
            return True
        except (OSError, ValueError) as exc:
            self.error = 'Recording failed: '+str(exc)
            self.active = False
            return False

    def frame(self, snapshot, *, initial=False):
        elapsed = 0. if initial else max(perf_counter()-self.started, self.last_elapsed+1e-9)
        if self._write(self.frames, dict(elapsed_s=elapsed, frame=json_value(snapshot.frame), error=snapshot.error)):
            self.last_elapsed = elapsed
            self.frame_count += 1

    def event(self, kind, **values):
        if self._write(self.events, dict(elapsed_s=perf_counter()-self.started, kind=kind, **values)):
            self.event_count += 1

    def close(self):
        if self.frames is None:
            return self.manifest
        self.active = False
        for handle in (self.frames, self.events):
            try:
                handle.flush()
                os.fsync(handle.fileno())
            except OSError as exc:
                self.error = 'Recording save failed: '+str(exc)
            finally:
                try:
                    handle.close()
                except OSError as exc:
                    self.error = 'Recording close failed: '+str(exc)
        self.frames = self.events = None
        self.metadata.update(complete=self.error is None, error=self.error,
            frame_count=self.frame_count, event_count=self.event_count, duration_s=self.last_elapsed,
            stopped_utc=datetime.now(timezone.utc).isoformat())
        try:
            self._manifest()
        except OSError as exc:
            self.error = 'Recording manifest save failed: '+str(exc)
        return self.manifest


class RecordedDisplay:
    """Index frame offsets once; seek reads only the requested frame from disk."""
    def __init__(self, path):
        path = Path(path).expanduser().resolve()
        self.path = path/'recording.json' if path.is_dir() else path
        metadata = json.loads(self.path.read_text())
        if metadata.get('schema') != SCHEMA:
            raise ValueError('Select an AeroWhip display recording.json file')
        self.metadata = metadata
        self.configuration = decode_configuration(metadata['configuration'])
        if metadata.get('frames') != 'frames.jsonl':
            raise ValueError('Unsupported recording frames file')
        self.file = self.path.with_name('frames.jsonl').open('rb')
        self.warning = None if metadata.get('complete') else 'Incomplete recording: replaying retained complete frames'
        self.offsets, times = [], []
        try:
            while True:
                offset = self.file.tell()
                line = self.file.readline()
                if not line:
                    break
                try:
                    row = json.loads(line)
                    elapsed = float(row['elapsed_s'])
                    if not math.isfinite(elapsed) or elapsed < 0 or (times and elapsed <= times[-1]):
                        raise ValueError('Recording times must be finite and strictly increasing')
                    decode_frame(row['frame'])
                except (ValueError, KeyError, TypeError):
                    # Only an interrupted final line may be skipped, and only
                    # for a manifest that never claimed a completed recording.
                    if not metadata.get('complete') and not line.endswith(b'\n') and not self.file.read(1):
                        break
                    raise
                self.offsets.append(offset)
                times.append(elapsed)
            if not times or times[0] != 0.:
                raise ValueError('Recording has no usable initial frame')
            if metadata.get('complete') and metadata.get('frame_count') != len(times):
                raise ValueError('Completed recording frame count mismatch')
            self.time = np.asarray(times)
            self.speed = 1.
            self.reset()
        except Exception:
            self.file.close()
            raise

    @property
    def duration_s(self):
        return float(self.time[-1])

    def seek(self, index):
        if not 0 <= index < len(self.time):
            raise IndexError('Recording sample index out of range')
        self.index = int(index)
        self._cursor_s = float(self.time[self.index])
        self.file.seek(self.offsets[self.index])
        row = json.loads(self.file.readline())
        self.frame = replace(decode_frame(row['frame']), recorded=True,
            source_label='Recorded · '+row['frame']['source_label'])
        self.error = row.get('error')

    def reset(self):
        self.running = False
        self.seek(0)

    def toggle_running(self):
        if not self.running and self.index == len(self.time)-1:
            self.seek(0)
        self.running = not self.running

    def step(self):
        self.running = False
        self.seek(min(self.index+1, len(self.time)-1))

    def advance(self, elapsed_s):
        if not math.isfinite(elapsed_s) or elapsed_s < 0:
            raise ValueError('Invalid replay elapsed time')
        if not self.running:
            return 0
        cursor = min(self.duration_s, self._cursor_s+elapsed_s*self.speed)
        index = max(0, int(np.searchsorted(self.time, cursor, side='right')-1))
        changed = index != self.index
        if changed:
            self.seek(index)
        self._cursor_s = cursor
        if cursor >= self.duration_s:
            self.running = False
        return int(changed)


class RecordingSource(ReplaySource):
    """Reuse the viewer playback controls with saved mixed-provenance frames."""
    can_replay_errors = True

    def __init__(self, path):
        replay = RecordedDisplay(path)
        super().__init__(replay, replay.configuration)

    def _publish(self):
        self._revision += 1
        message = '; '.join(value for value in (self.replay.warning, self.replay.error) if value) or None
        self.latest = SourceSnapshot(self._revision, None, self.replay.frame,
            self.replay.running, message, self.replay.index, self.replay.speed, perf_counter())

    @property
    def playback_time_s(self):
        return float(self.replay.time[self.replay.index])

    def close(self):
        super().close()
        self.replay.file.close()
