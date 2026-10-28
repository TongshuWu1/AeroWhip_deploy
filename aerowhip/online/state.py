"""The running model: the online estimate carried from flight to flight (user's decision 2026-10-28:
apply the online fit straight away, old history discounted with a 5-minute half-life of flight).

runs/online/model_state.json holds the six online parameters, their covariance, the offline prior
they started from, and the history of the flights that updated them. `online_fit.py` starts from it
and writes it after every window of a real flight (flights under the flight stack's calibration_data;
replays and synthetic flights leave it alone unless given --state). The planner, the ensemble and
the 3D preview read it through `planning_fits`: the offline fits with the running values and
uncertainties put in, except the command delay (held at the offline fit: the online delay is in the
replay's timing, one frame off the closed-loop simulator's) and with any value more than
MAX_PLANNING_SD offline standard deviations from the offline fit held at that limit and reported.
"""
from copy import deepcopy
from datetime import datetime
import json
import math
import os
from pathlib import Path

import numpy as np

from aerowhip.paths import ROOT
from aerowhip.planning.simulate import FLIGHT_STACK

STATE = ROOT/'runs'/'online'/'model_state.json'
REAL_DATA = FLIGHT_STACK/'calibration_data'
MAX_PLANNING_SD = 4.
HELD = ('command_delay_s',)          # not passed to the planner (see the module docstring)


def load(path=None):
    try:
        return json.loads(Path(path or STATE).read_text())
    except (OSError, ValueError):
        return None


def save(path, estimator, *, flight=None, finished=False):
    """Write the estimator's state (atomically); on `finished`, add the flight to the history."""
    path = Path(path)
    old = load(path) or {}
    history = old.get('flights', [])
    names = estimator.space.names
    entry = dict(flight=str(flight), finished=datetime.now().isoformat(timespec='seconds'),
                 theta={n: float(v) for n, v in zip(names, estimator.theta)},
                 counts={f: dict(s.counts) for f, s in estimator.families.items()})
    data = dict(parameters=names, theta=[float(v) for v in estimator.theta], P=np.asarray(estimator.P).tolist(),
                prior={p.name: dict(value=p.value, sd=p.sd) for p in estimator.space.parameters},
                half_life_s=estimator.half_life, updated=datetime.now().isoformat(timespec='seconds'),
                current_flight=None if finished else str(flight), flights=history+([entry] if finished else []))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=1)+'\n')
    os.replace(tmp, path)


def reject_flight(path, before, flight, reason):
    """A failed take (the user's rule: the cable touched the drone, or the take was interrupted) is not
    used at all: the running model goes back to its state from before the flight (`before`, the file's
    content then, or None if there was none), and the flight is listed in the history as rejected."""
    path = Path(path)
    entry = dict(flight=str(flight), finished=datetime.now().isoformat(timespec='seconds'), rejected=reason)
    if before is None:
        rejected = path.with_name('model_state_rejected.jsonl')
        with rejected.open('a') as f:
            f.write(json.dumps(entry)+'\n')
        if path.exists():
            path.unlink()                          # it only held this flight's estimate
        return
    data = dict(before, flights=before.get('flights', [])+[entry], current_flight=None,
                updated=datetime.now().isoformat(timespec='seconds'))
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=1)+'\n')
    os.replace(tmp, path)


def start_from(estimator, state):
    """Start the estimator from a saved running model (same parameters), else leave the prior."""
    if not state or state.get('parameters') != estimator.space.names:
        return False
    estimator.theta = np.asarray(state['theta'], float)
    estimator.P = np.asarray(state['P'], float)
    return True


def reset(path=None):
    """Back to the offline fits: the running model is kept as model_state_<time>.json, not deleted."""
    path = Path(path or STATE)
    if path.exists():
        os.replace(path, path.with_name(f'model_state_{datetime.now():%Y%m%d-%H%M%S}.json'))


def is_real_flight(folder):
    try:
        Path(folder).resolve().relative_to(REAL_DATA.resolve())
        return True
    except ValueError:
        return False


def planning_values(state):
    """{name: (value used for planning, note or None)} from a running model, with the holds applied."""
    out = {}
    for k, name in enumerate(state['parameters']):
        prior = state['prior'][name]
        if name in HELD:
            out[name] = (prior['value'], 'held at the offline fit (timing convention)')
            continue
        v = float(state['theta'][k])
        lo, hi = prior['value']-MAX_PLANNING_SD*prior['sd'], prior['value']+MAX_PLANNING_SD*prior['sd']
        note = None
        if not lo <= v <= hi:
            note = f'{v:.4g} is {abs(v-prior["value"])/prior["sd"]:.1f} offline sd from the offline fit: held at {MAX_PLANNING_SD:g} sd for planning'
            v = min(max(v, lo), hi)
        out[name] = (v, note)
    return out


def planning_fits(cable_fit, drone_fit, state=None):
    """(cable_fit, drone_fit, notes): the offline fits with the running model's values and uncertainties
    for the online parameters (none of it if there is no running model). The planner's ensemble and
    the preview take these in place of the archived fits."""
    if state is None:
        return cable_fit, drone_fit, ['no running model: the offline fits']
    cable_fit, drone_fit = deepcopy(cable_fit), deepcopy(drone_fit)
    values = planning_values(state)
    names = state['parameters']
    P = np.asarray(state['P'], float)
    sd = {n: math.sqrt(max(P[k, k], 0.)) for k, n in enumerate(names)}
    notes = [f'running model of {state.get("updated", "?")} ({len(state.get("flights", []))} flights)']
    for fit in (cable_fit, drone_fit):
        params = fit['parameters']
        u = fit['uncertainty']
        u['standard_deviation'] = list(u['standard_deviation'])
        corr = np.asarray(u.get('correlation', np.eye(len(params))), float).copy()
        mine = [n for n in names if n in params and n not in HELD]
        for n in mine:
            i = params.index(n)
            fit['best']['x'][i] = values[n][0]
            u['standard_deviation'][i] = sd[n]
            corr[i, :] = 0.; corr[:, i] = 0.; corr[i, i] = 1.
        for a in mine:                           # the running model's correlations within the family
            for b in mine:
                if a != b:
                    ia, ib = names.index(a), names.index(b)
                    corr[params.index(a), params.index(b)] = P[ia, ib]/max(math.sqrt(P[ia, ia]*P[ib, ib]), 1e-300)
        u['correlation'] = corr.tolist()
    notes += [f'{n}: {note}' for n, (_, note) in values.items() if note]
    return cable_fit, drone_fit, notes
