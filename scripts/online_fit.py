"""Online identification of a flight: the moving-window estimator (aerowhip.online) on a flight
folder, live while it is written or replayed from a recording as if live.

    # replay a recorded flight as if live (as fast as the computer can; --speed 1 for real time)
    python scripts/online_fit.py --flight <calibration_data>/cf_6/trajectory/001 --replay
    # live: follow the next flight folder that appears under the setup's data (start before take-off)
    python scripts/online_fit.py --watch <calibration_data>/cf_6
    # validation on cf_2's recordings with cf_2's model, from a deliberately wrong start
    python scripts/online_fit.py --flight .../archive/cf_2_2026-10-28/cf_2/strokes/001 --replay \
        --model-commit 45900ad --start "log10 EI=+0.3, command_delay_s=+0.010, motor_response_s=*1.5"

Writes runs/online/<flight>-<time>.jsonl: a header line (flight, model, prior, settings), then one
record per update (the parameters and their standard deviations, and per family the status:
accepted / outlier / change / gated with the reason), and <...>.summary.json at the end. The
Identification Loop window plots the newest one live.

During a real flight this runs on the flight computer: it lowers its own priority and can be pinned
to cores (--cpus) the flight stack does not use; it reads files only (no ROS).
"""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'scripts'))
from aerowhip.online.estimator import ModelSpace, OnlineEstimator  # noqa: E402
from aerowhip.online import state as running  # noqa: E402
from aerowhip.online.stream import FlightStream  # noqa: E402
from aerowhip.physics.model_configuration import DEFAULT_MODEL, fit_configuration, load_model_configuration  # noqa: E402
from aerowhip.planning.ensemble import load_fit  # noqa: E402
from plan_flight import ARCHIVE  # noqa: E402

OUTPUT = ROOT/'runs'/'online'


def base_model(commit=None):
    """The model configuration now, or the one at a git commit (e.g. cf_2's, 45900ad)."""
    if commit is None:
        return load_model_configuration(DEFAULT_MODEL)
    def at(path):
        return json.loads(subprocess.run(['git', 'show', f'{commit}:{path}'], cwd=ROOT, check=True, capture_output=True, text=True).stdout)
    files = at('configs/simulation.json')
    roles = {}
    for role, path in files.items():
        roles[role] = at(path) if path.endswith('.json') else path   # .toml roles are read from the tree
    return load_model_configuration(roles)


def parse_start(text, space):
    """'name=+d, name=*k, name=v' -> a start vector (offset, factor or value)."""
    theta = np.array([p.value for p in space.parameters], float)
    for item in filter(None, (s.strip() for s in (text or '').split(','))):
        name, value = (s.strip() for s in item.split('='))
        k = space.names.index(name)
        if value[0] == '+':                 # '+d' offset ('+-d' lowers)
            theta[k] += float(value[1:])
        elif value[0] == '*':               # '*k' factor
            theta[k] *= float(value[1:])
        else:
            theta[k] = float(value)
    return theta


def newest_flight(root, after):
    """The newest flight folder (<take>/<NNN> with parameters.json) under root created after `after`."""
    found = [p.parent for p in Path(root).glob('*/*/parameters.json') if p.stat().st_mtime > after]
    return max(found, key=lambda p: p.stat().st_mtime) if found else None


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument('--flight', type=Path, help='a flight folder (live unless --replay)')
    src.add_argument('--watch', type=Path, help='wait for the next flight folder under this setup folder, then follow it live')
    p.add_argument('--replay', action='store_true', help='replay a recorded flight as if live')
    p.add_argument('--speed', type=float, default=0., help='replay speed (1 = real time; 0 = as fast as possible)')
    p.add_argument('--period', type=float, default=1., help='seconds of flight between updates')
    p.add_argument('--window', type=float, default=2.)
    p.add_argument('--threads', type=int, default=8)
    p.add_argument('--cpus', help="pin to these cores, e.g. 8-15, or 'auto' (the upper half): keep the flight stack's cores free")
    p.add_argument('--model-commit', help='use the model configuration at this git commit (cf_2: 45900ad)')
    p.add_argument('--cable-fit', type=Path, default=ARCHIVE/'cable'/'20261028-012011.json')
    p.add_argument('--drone-fit', type=Path, default=ARCHIVE/'drone'/'20261027-201939.json')
    p.add_argument('--start', help='start away from the prior mean, e.g. "log10 EI=+0.3, motor_response_s=*1.5"')
    p.add_argument('--families', default='cable,drone')
    p.add_argument('--output', type=Path)
    p.add_argument('--state', default='auto', help="the running model to start from and update: 'auto' (runs/online/model_state.json "
                   "for real flights followed live; none for replays and synthetic flights), a path, or 'none'")
    p.add_argument('--quiet', action='store_true')
    args = p.parse_args()
    os.nice(10)
    if args.cpus:                       # 'auto': the upper half of the cores (the flight stack and its radio
        if args.cpus == 'auto':         # link run on the host's CPU; a loaded host delays the radio)
            n = os.cpu_count() or 2
            cores = set(range(n//2, n))
        else:
            lo, _, hi = args.cpus.partition('-')
            cores = set(range(int(lo), int(hi or lo)+1))
        os.sched_setaffinity(0, cores)

    base = base_model(args.model_commit)
    space = ModelSpace(base, load_fit(args.cable_fit), load_fit(args.drone_fit))
    estimator = OnlineEstimator(space, window_s=args.window, threads=args.threads, families=tuple(args.families.split(',')))
    if args.start:
        estimator.theta = parse_start(args.start, space)

    folder = args.flight
    if args.watch:
        print(f'waiting for the next flight under {args.watch} ...', flush=True)
        started = time.time()
        while (folder := newest_flight(args.watch, started)) is None:
            time.sleep(.5)
    cfg = fit_configuration(base)
    arc = np.cumsum(cfg['cable']['marker_interval_lengths_m'])
    attachment = cfg['attachment']['position_tracking_m']
    stream = FlightStream(folder, arc, attachment, replay=args.replay)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    out = args.output or OUTPUT/f'{folder.parent.name}_{folder.name}-{datetime.now():%Y%m%d-%H%M%S}.jsonl'
    log = out.open('w', buffering=1)
    log.write(json.dumps(dict(kind='header', flight=str(folder), live=not args.replay, model_commit=args.model_commit,
                              cable_fit=str(args.cable_fit), drone_fit=str(args.drone_fit), window_s=args.window, period_s=args.period,
                              parameters=space.names, prior=estimator.summary()['prior'],
                              start={n: float(v) for n, v in zip(space.names, estimator.theta)}))+'\n')
    print(f'{folder} -> {out}', flush=True)
    state_path = None
    if args.state == 'auto':
        state_path = running.STATE if (not args.replay and running.is_real_flight(folder)) else None
    elif args.state != 'none':
        state_path = Path(args.state)
    before = None
    if state_path is not None and not args.start:
        before = running.load(state_path)
        started = running.start_from(estimator, before)
        print(f"running model: {state_path} ({'continued' if started else 'new, from the offline fits'})", flush=True)
        log.write(json.dumps(dict(kind='state', path=str(state_path), continued=started,
                                  start={n: float(v) for n, v in zip(space.names, estimator.theta)}))+'\n')

    unusable, records = [], []

    def update():
        try:
            flight = stream.flight()
        except ValueError as exc:            # e.g. a capture rate the windows cannot use: say it once
            if not unusable:
                unusable.append(str(exc))
                log.write(json.dumps(dict(kind='error', stamp=stream.latest, reason=str(exc)))+'\n')
                print('cannot use this flight:', exc, flush=True)
            return
        if flight is None:
            return
        record = estimator.step(flight, stream.params, has_cable=stream.has_cable_labels)
        record['kind'] = 'update'
        records.append(record)
        if state_path is not None:
            running.save(state_path, estimator, flight=folder)
        log.write(json.dumps(record)+'\n')
        if not args.quiet:
            fams = '  '.join(f"{n}: {r['status']}{' ('+r['reason']+')' if r.get('reason') and r['status'] != 'accepted' else ''}"
                             for n, r in record['families'].items())
            vals = '  '.join(f"{n.split(' ')[-1][:10]} {record['theta'][n]:.4g}±{record['sd'][n]:.2g}" for n in space.names)
            print(f"{record['stamp']-t_start:6.1f} s  {record['phase'][:14]:14s} {fams}\n         {vals}   [{record['seconds']:.2f} s]", flush=True)

    t_start = None
    if args.replay:
        stamps = [r['stamp'] for r in stream.queue['poses.jsonl']]
        t_start = stamps[0]
        wall = time.monotonic()
        t = t_start+args.window+.2
        while t <= stream.end:
            stream.advance(t)
            update()
            t += args.period
            if args.speed > 0:
                time.sleep(max(0., wall+(t-t_start)/args.speed-time.monotonic()))
    else:
        quiet_since = None
        next_at = None
        while True:
            stream.advance()
            if t_start is None and stream.poses:
                t_start = stream.poses[0]['stamp']
            if stream.latest > -np.inf:
                if next_at is None:
                    next_at = stream.latest+args.window+.2
                if stream.latest >= next_at:
                    update()
                    next_at = stream.latest+args.period
            grew = any(len(q) for q in stream.queue.values())
            if stream.finished and not grew:
                quiet_since = quiet_since or time.monotonic()
                if time.monotonic()-quiet_since > 2.:
                    break
            time.sleep(.05)
    failed = None                    # the user's rule: a take that failed is not used at all
    result = running.load(folder/'result.json')
    if result is not None and not result.get('completed'):
        failed = f"take failed: {result.get('failure') or 'not completed'}"
    touched = [r for r in records if any(str(f.get('reason', '')).startswith('cable touching') for f in r['families'].values())]
    if touched:
        failed = f"cable touched the drone at {touched[0]['stamp']-t_start:.1f} s"
    if state_path is not None:
        if failed:
            running.reject_flight(state_path, before, folder, failed)
            print(f'running model: flight rejected ({failed}); back to the state before it', flush=True)
        else:
            running.save(state_path, estimator, flight=folder, finished=True)
    log.write(json.dumps(dict(kind='verdict', used=failed is None, reason=failed))+'\n')
    summary = estimator.summary()
    estimator.close()
    log.write(json.dumps(dict(kind='summary', **summary))+'\n')
    log.close()
    out.with_suffix('.summary.json').write_text(json.dumps(summary, indent=1)+'\n')
    print('\nfinal:')
    for n in space.names:
        pr = summary['prior'][n]
        print(f"  {n:22s} {summary['theta'][n]:10.5g} ± {summary['sd'][n]:.3g}   (prior {pr['value']:.5g} ± {pr['sd']:.3g})")
    print('  windows:', {f: c for f, c in summary['counts'].items()})


if __name__ == '__main__':
    main()
