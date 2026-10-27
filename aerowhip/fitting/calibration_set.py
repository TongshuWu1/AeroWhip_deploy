"""The 2026-10-27 calibration flights as a fitting data set: usable spans and the split.

Usable spans come from the take log (`takes.jsonl`, last entry per folder): `exclude_phases`
and `usable_until_t_s`. The whip flights were flown as validation and, with random_high, are
held out of every fit; they are only scored with the final parameters.
"""
import json
from pathlib import Path

import numpy as np

from aerowhip.takes.calibration_flights import load, windows

DEFAULT_ROOT = Path('/home/tony-lab/Documents/PHD/Crazyflie/crazyswarm2_jazzy_whipping/calibration_data')
SESSION = '2026-10-27'
MOTIONS = ('circle', 'decay', 'figure8', 'line_star', 'random', 'random_high', 'strokes', 'vertical', 'whip')
HELD_OUT = ('random_high', 'whip')
# Spans the take log does not mark: the drone-only strokes flight ran its battery to 6.78 V
# under load in 'strokes 90 4' and lost the link in 'strokes 90 6' (11 m replay error there).
USABLE_UNTIL_T_S = {f'strokes/{SESSION}_cf_2_nocable_strokes_01': 104.1}


def take_notes(root=DEFAULT_ROOT):
    notes = {}
    for line in (Path(root)/'takes.jsonl').read_text().splitlines():
        entry = json.loads(line)
        notes[entry['folder']] = entry      # last entry per folder wins
    return notes


def folders(setup, root=DEFAULT_ROOT):
    """{motion: folder} for one setup ('cf_2' with cable, 'cf_2_nocable' without)."""
    root = Path(root)
    out = {}
    for motion in MOTIONS:
        hits = sorted(root.glob(f'*/{SESSION}_{setup}_{motion}_0*'))
        hits = [h for h in hits if h.is_dir() and (h/'poses.jsonl').exists()]
        if hits:
            out[motion] = hits[-1] if motion != 'vertical' else hits[0]
    return out


def flight_windows(folder, *, root=DEFAULT_ROOT, duration_s=2., stride_s=1., min_seen=8, cable=True):
    """Windows of one flight inside its usable span."""
    flight = load(folder)
    if not cable:
        flight.markers = None
    note = take_notes(root).get(str(Path(folder).relative_to(root)), {})
    excluded = tuple(note.get('exclude_phases') or ())
    until = note.get('usable_until_t_s', USABLE_UNTIL_T_S.get(str(Path(folder).relative_to(root))))
    result = []
    for w in windows(flight, duration_s=duration_s, stride_s=stride_s, min_seen=min_seen):
        if w.phase.startswith(('takeoff', 'land', 'settle')) or any(w.phase.startswith(p) for p in excluded):
            continue
        if until is not None:
            # flight time of the window end, from the log's stamp/time pairing
            ok = np.isfinite(flight.log_stamp)
            t_end = np.interp(w.start_stamp+w.time[-1], flight.log_stamp[ok], flight.log_time[ok])
            if t_end > until:
                continue
        result.append(w)
    return result


def split(setup, *, root=DEFAULT_ROOT, cable=True, **kwargs):
    """({motion: windows} for fitting, {motion: windows} held out)."""
    train, held = {}, {}
    for motion, folder in folders(setup, root).items():
        target = held if motion in HELD_OUT else train
        target[motion] = flight_windows(folder, root=root, cable=cable, **kwargs)
    return train, held
