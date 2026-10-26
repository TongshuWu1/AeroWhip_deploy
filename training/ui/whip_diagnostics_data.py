"""Read-only saved-motion diagnostics; no task scoring or simulation execution."""
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np


def recording_choices(directory, history):
    """Only saved individual motions, never a padded batch mistaken for a replay."""
    directory = Path(directory)
    choices, seen = [], set()
    for row in reversed(history):
        name = row.get('replay')
        if row.get('phase') not in ('evaluation', 'curriculum_sample') or not name:
            continue
        if name in seen or Path(name).name != name:
            continue
        seen.add(name)
        label = f"Evaluation · collection {row.get('round', '?')}"
        if row.get('phase') == 'curriculum_sample':
            label = f"Saved curriculum sample · collection {row.get('round', '?')}"
        choices.append((label, name))
    for path in sorted(directory.glob('collection-*-motions.json'), reverse=True):
        try:
            samples = json.loads(path.read_text()).get('samples', [])
        except (OSError, ValueError, AttributeError):
            continue  # A writer may still be publishing this manifest.
        collection = path.name.split('-')[1].lstrip('0') or '0'
        for row in samples:
            name = row.get('replay')
            if not isinstance(name, str) or Path(name).name != name or name in seen:
                continue
            seen.add(name)
            outcome = 'invalid' if row.get('valid') is False else 'valid' if row.get('valid') is True else 'validity unknown'
            choices.append((f"Training sample · collection {collection} · lane {row.get('lane', '?')} · {outcome}", name))
    return choices


@dataclass
class WhipDiagnosticsData:
    path: Path
    time: np.ndarray
    positions: np.ndarray
    relative_positions: np.ndarray
    velocities: np.ndarray
    relative_velocities: np.ndarray
    rest: np.ndarray
    reach: np.ndarray
    tip_height: np.ndarray
    world_speed: np.ndarray
    relative_speed: np.ndarray
    world_outward: np.ndarray
    relative_outward: np.ndarray
    world_cosine: np.ndarray
    relative_cosine: np.ndarray
    joint_turn_deg: np.ndarray
    density: np.ndarray | None
    grid: np.ndarray | None
    regions: list
    record: dict
    events: list
    mode: str
    validity: bool | None
    notices: list
    continuous: dict

    @property
    def length(self):
        return float(self.rest.sum())

    @classmethod
    def load(cls, directory, filename):
        directory = Path(directory).resolve()
        path = (directory / filename).resolve()
        if Path(filename).name != filename or path.parent != directory:
            raise ValueError('Choose a saved NPZ inside the selected run.')
        snapshot = json.loads((directory / 'configuration.json').read_text())
        with np.load(path, allow_pickle=False) as archive:
            required = ('time_s', 'cable_positions', 'cable_velocities')
            if any(key not in archive for key in required):
                raise ValueError('This recording has no saved cable positions/velocities.')
            time, positions, velocities = [np.asarray(archive[key], dtype=float) for key in required]
            record, notices = {}, []
            if 'fold_event_json' in archive:
                value = archive['fold_event_json'].item()
                record = json.loads(value)
                if not isinstance(record, dict):
                    raise ValueError('Expected an individual saved fold record.')
            if 'physical_violation' in record:
                notices.append('Tilt and cable-height limits are diagnostic only; full horizon is not physical validity.')
                if record.get('physical_violation'):
                    notices.append(f"Physical envelope violated: minimum cable height {record['minimum_cable_height_m']:.3f} m; maximum tilt {record['maximum_tilt_deg']:.1f} degrees. No extra physical-violation reward penalty.")
            terminal = float(archive['terminal_time_s'].item()) if 'terminal_time_s' in archive else record.get('terminal_time_s')
            validity = bool(archive['task_valid'].item()) if 'task_valid' in archive else None
            valid_frames = np.asarray(archive['state_valid'], bool) if 'state_valid' in archive else None
            continuous={}
            if 'continuous_whip_diagnostics' in archive and 'continuous_whip_fields' in archive:
                values=np.asarray(archive['continuous_whip_diagnostics'],float)
                fields=archive['continuous_whip_fields'].tolist()
                if values.shape!=(len(time),len(fields)):
                    raise ValueError('Continuous reward trace does not match saved frames.')
                continuous={str(key):values[:,i] for i,key in enumerate(fields)}
        if positions.ndim != 3 or positions.shape[-1] != 3 or positions.shape[1] < 3:
            raise ValueError('Select an individual replay, not a batched collection archive.')
        if velocities.shape != positions.shape or time.shape != (len(positions),) or not len(time):
            raise ValueError('Saved position, velocity and time arrays disagree.')
        if not np.isfinite(time).all() or np.any(np.diff(time) <= 0):
            raise ValueError('Saved times must be finite and strictly increasing.')
        count = len(time)
        if terminal is not None:
            if not np.isfinite(terminal) or terminal < time[0]:
                raise ValueError('Invalid terminal time in recording.')
            count = min(count, int(np.searchsorted(time, terminal + 1e-9, side='right')))
        good = np.isfinite(positions).all(axis=(1, 2)) & np.isfinite(velocities).all(axis=(1, 2))
        if valid_frames is not None:
            if valid_frames.shape != time.shape:
                raise ValueError('Saved validity mask has the wrong shape.')
            good &= valid_frames
        bad = np.flatnonzero(~good[:count])
        if len(bad):
            count = int(bad[0])
        if count == 0:
            raise ValueError('No finite valid saved prefix to display.')
        if count < len(time):
            notices.append(f'Displaying {count}/{len(time)} saved frames; padded/invalid suffix omitted.')
        time, positions, velocities = time[:count], positions[:count], velocities[:count]
        continuous={key:values[:count] for key,values in continuous.items()}
        rest = np.asarray(snapshot.get('model', {}).get('cable', {}).get('marker_interval_lengths_m', []), float)
        if rest.shape != (positions.shape[1]-1,) or not np.isfinite(rest).all() or np.any(rest <= 0):
            raise ValueError('Saved model rest lengths do not match this recording; no guessed normalization.')
        rel = positions - positions[:, :1]
        rv = velocities - velocities[:, :1]
        ws = np.linalg.norm(velocities[:, -1], axis=1)
        rs = np.linalg.norm(rv[:, -1], axis=1)
        # Same fixed world +X output as the Stage 1 reward; zero speed has zero cosine.
        reach = np.maximum(0., rel[:, -1, 0])
        uw = velocities[:, -1, 0].copy()
        ur = rv[:, -1, 0].copy()
        wc = np.divide(uw, ws, out=np.zeros_like(ws), where=ws > 0)
        rc = np.divide(ur, rs, out=np.zeros_like(rs), where=rs > 0)
        edges = np.diff(positions, axis=1)
        edge_lengths = np.linalg.norm(edges, axis=2)
        if np.any(edge_lengths <= 0):
            raise ValueError('Collapsed segment in the saved geometry.')
        tangents = edges / edge_lengths[..., None]
        turn = np.arctan2(np.linalg.norm(np.cross(tangents[:, :-1], tangents[:, 1:]), axis=2),
                          np.sum(tangents[:, :-1] * tangents[:, 1:], axis=2))
        density, grid, regions = None, None, [[] for _ in time]
        if len(rest) == 10:
            from tasks.whip.horizontal_fold import FoldGeometry
            geometry = FoldGeometry(rest)
            density = turn @ geometry.kernel
            grid = geometry.grid.copy()
            regions = [geometry.observe(frame)[1] for frame in positions]
            notices.append('Bend field/track IDs reconstructed with current FoldGeometry; not historical detector outcomes.')
        else:
            notices.append('Bend tracking unavailable for this mesh; showing raw joint turns only.')
        events = []
        for event in record.get('events', []):
            when = event.get('time_s')
            if isinstance(when, (int, float)) and np.isfinite(when) and event.get('kind') != 'progress':
                events.append(event)
        if not record:
            notices.append('No saved fold-event log. Detector phase and qualification are unavailable; no rescoring performed.')
        return cls(path, time, positions, rel, velocities, rv, rest,
                   reach/rest.sum(), rel[:, -1, 2], ws, rs, uw, ur, wc, rc,
                   np.degrees(turn), density, grid, regions, record, events,
                   snapshot.get('task', {}).get('continuous_whip', {}).get('version', 'unknown'), validity, notices,continuous)

    def phase(self, index):
        past = [e for e in self.events if e['time_s'] <= self.time[index]+1e-9]
        if not self.record:
            return 'No recorded fold phase'
        if not past:
            return 'No recorded fold milestone yet'
        labels = {'armed': 'Fold armed', 'propagated': 'Propagation recorded',
                  'opening_confirmed': 'Opening recorded', 'event_end': 'Output phase ended',
                  'hit': 'Recorded hit', 'timeout': 'Deadline reached', 'discarded': 'Candidate discarded'}
        kind = past[-1]['kind']
        return labels.get(kind, kind.replace('_', ' '))
