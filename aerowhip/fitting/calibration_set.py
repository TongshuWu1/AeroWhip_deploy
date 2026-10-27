"""The calibration flights as a fitting data set: which flights, their roles and usable spans.

All of it comes from the calibration manifest (`model_calibration.yaml` in the data root, see
`aerowhip.fitting.manifest`). Flights are organised by category (setup and take) and pooled across
the days they were flown: one model for all of them, with the per-flight conditions that are
measured (battery voltage) as inputs. Role 'fit' flights are fitted, 'held_out' flights are scored
only with the final parameters, 'hover' flights serve the motor identification, 'excluded' ones are
not used. Usable spans (`usable.until_t_s`, `usable.exclude_phases`) trim a flight.
"""
from pathlib import Path

import numpy as np

from aerowhip.fitting.manifest import Manifest
from aerowhip.takes.calibration_flights import load, windows

DEFAULT_ROOT = Path('/home/tony-lab/Documents/PHD/Crazyflie/crazyswarm2_jazzy_whipping/calibration_data')


def manifest(root=DEFAULT_ROOT):
    return Manifest(root)


def named(entries):
    """{name: entry}: the take name when a category has one flight, else take/NNN."""
    count = {}
    for f in entries:
        count[f['motion']] = count.get(f['motion'], 0)+1
    return {(f['motion'] if count[f['motion']] == 1 else f"{f['motion']}/{Path(f['folder']).name}"): f for f in entries}


def folders(setup, root=DEFAULT_ROOT, roles=('fit', 'held_out')):
    """{name: folder} of every flight of one setup ('cf_2' with cable, 'cf_2_nocable' without)."""
    m = manifest(root)
    return {name: m.folder(f) for name, f in named(m.flights(setup, roles=roles)).items()}


def held_out(setup='cf_2', root=DEFAULT_ROOT):
    """Names (as in `folders`) of the flights held out of every fit."""
    m = manifest(root)
    return tuple(name for name, f in named(m.flights(setup)).items() if f['role'] == 'held_out')


def is_held_out(folder, root=DEFAULT_ROOT):
    m = manifest(root)
    return any(m.folder(f) == Path(folder) and f['role'] == 'held_out' for f in m.data['flights'])


def flight_windows(folder, *, root=DEFAULT_ROOT, duration_s=2., stride_s=1., min_seen=8, cable=True):
    """Windows of one flight inside its usable span."""
    flight = load(folder)
    if not cable:
        flight.markers = None
    elif flight.markers is None and (Path(folder)/'cloud.jsonl').exists():
        # a cable flight without offline labels would give marker-less windows that every replay fails on
        raise ValueError(f'{folder}: cable markers not labelled yet (run scripts/label_sequence.py on it)')
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
    """({name: windows} for fitting, {name: windows} held out)."""
    train, held = {}, {}
    m = manifest(root)
    for name, f in named(m.flights(setup)).items():
        target = held if f['role'] == 'held_out' else train
        target[name] = flight_windows(m.folder(f), root=root, cable=cable, **kwargs)
    return train, held
