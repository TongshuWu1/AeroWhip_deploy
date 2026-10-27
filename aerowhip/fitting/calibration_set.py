"""The calibration flights as a fitting data set: which flights, their roles and usable spans.

All of it comes from the calibration manifest (`model_calibration.yaml` in the data root, see
`aerowhip.fitting.manifest`): flights with role 'fit' are fitted, 'held_out' flights are scored
only with the final parameters, 'hover' flights serve the motor identification, 'excluded' ones
are not used. Usable spans (`usable.until_t_s`, `usable.exclude_phases`) trim a flight.
"""
from pathlib import Path

import numpy as np

from aerowhip.fitting.manifest import Manifest
from aerowhip.takes.calibration_flights import load, windows

DEFAULT_ROOT = Path('/home/tony-lab/Documents/PHD/Crazyflie/crazyswarm2_jazzy_whipping/calibration_data')
SESSION = '2026-10-27'


def manifest(root=DEFAULT_ROOT):
    return Manifest(root)


def held_out(setup='cf_2', root=DEFAULT_ROOT, session=SESSION):
    """Motions held out of every fit for this setup."""
    return tuple(f['motion'] for f in manifest(root).flights(setup, roles=('held_out',), session=session))


HELD_OUT = ('random_high', 'whip')     # the 2026-10-27 split; `held_out()` reads the manifest


def folders(setup, root=DEFAULT_ROOT, session=SESSION, roles=('fit', 'held_out')):
    """{motion: folder} for one setup ('cf_2' with cable, 'cf_2_nocable' without)."""
    m = manifest(root)
    out = {}
    for f in m.flights(setup, roles=roles, session=session):
        if f['motion'] in out:
            raise ValueError(f"{setup}: two {f['motion']} flights with a fitting role in session {session}; mark one 'excluded'")
        out[f['motion']] = m.folder(f)
    return out


def flight_windows(folder, *, root=DEFAULT_ROOT, duration_s=2., stride_s=1., min_seen=8, cable=True):
    """Windows of one flight inside its usable span."""
    flight = load(folder)
    if not cable:
        flight.markers = None
    m = manifest(root)
    entry = next((f for f in m.data['flights'] if m.folder(f) == Path(folder)), None)
    until, excluded = m.usable(entry) if entry else (None, ())
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
    m = manifest(root)
    for f in m.flights(setup, session=SESSION):
        target = held if f['role'] == 'held_out' else train
        target[f['motion']] = flight_windows(m.folder(f), root=root, cable=cable, **kwargs)
    return train, held
