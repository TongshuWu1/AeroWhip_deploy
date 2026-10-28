"""Plan the next calibration flight: the parts (motion primitives with levels and headings) that
are most informative and most new, that every member of the model ensemble predicts safe, and that
stay within one step of the measured envelope. Writes a plan file the flight script flies
(`acceleration_circle_cf2.py --plan FILE`) and a report of every candidate.

Per candidate part (kind, heading, level) and per ensemble member, the part is flown closed loop in
the simulator (aerowhip.planning.simulate). Then:
  constraints (every member unless --delta allows a share to fail): the cable's highest node stays
      below the attachment by --rise-margin, the drone's tilt below --tilt-limit, its position error
      below --error-limit, every motor above --motor-floor of its hover share, no engine failure;
  envelope: the level is at most --step above the highest level flown cleanly for that kind and
      heading (calibration_data/envelope/<setup>.json from scripts/envelope.py); with nothing flown
      yet, at most --first-level;
  score = disagreement (the members' spread of the cable tip path and the drone tilt over the part:
      what a flight there would teach) x novelty (1 for a (kind, heading, level) not flown cleanly
      yet, --flown-weight otherwise).
The plan takes the best feasible parts, at most one per (kind, heading), gentlest first, until the
motion budget is spent.

    python scripts/plan_flight.py --setup cf_6 --output runs/planning/plan_001.yaml
"""
import argparse
import itertools
import json
from pathlib import Path
import sys
import time
import tomllib

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from aerowhip.fitting import flight_drone
from aerowhip.fitting.calibration_set import DEFAULT_ROOT
from aerowhip.fitting.least_squares import ProcessRunner
from aerowhip.physics.model_configuration import DEFAULT_MODEL, fit_configuration, load_model_configuration
from aerowhip.planning.ensemble import BRACKETS, Ensemble, load_fit
from aerowhip.planning.simulate import FLIGHT_STACK, simulate_part

ARCHIVE = ROOT/'runs'/'archive_cf_2_2026-10-28'/'fits'
KINDS = dict(decay=dict(headings=True, sharp=True), strokes=dict(headings=True, sharp=True), whip=dict(headings=True, sharp=True),
             line=dict(headings=True, sharp=False), circle=dict(headings=True, sharp=False),
             figure8=dict(headings=False, sharp=False), vertical=dict(headings=False, sharp=False), random=dict(headings=False, sharp=False))


def flown_gains(stack):
    firmware = tomllib.loads(flight_drone.FIRMWARE.read_text())
    flown = yaml.safe_load((stack/'config'/'crazyflies.yaml').read_text())['all']['firmware_params'].get('ctrlMel', {})
    for key in ('kR_xy', 'kw_xy', 'kd_omega_rp', 'ki_m_xy', 'i_range_m_xy', 'kR_z', 'kw_z'):
        if key in flown:
            firmware['ctrlMel'][key] = float(flown[key])
    return firmware


def envelope_table(path):
    """{(kind, heading, level): summary} and the clean maximum level per (kind, heading)."""
    if not path.exists():
        return {}, {}
    data = json.loads(path.read_text())
    table, clean = {}, {}
    for s in data['summary']:
        key = (s['kind'], s['heading'], s['level'])
        table[key] = s
        if s['clean'] > 0 and s['level'] is not None:
            clean[(s['kind'], s['heading'])] = max(clean.get((s['kind'], s['heading']), 0.), s['level'])
    return table, clean


def candidates(kinds, headings, levels, clean, step, first_level, cable):
    out = []
    for kind in kinds:
        info = KINDS[kind]
        for heading in (headings if info['headings'] else [None]):
            allowed = clean.get((kind, heading), None)
            top = first_level if allowed is None else allowed+step
            for level in levels:
                if level > top+1e-9:
                    continue
                spec = dict(kind=kind, level=float(level))
                if heading is not None:
                    spec['heading'] = float(heading)
                if kind == 'decay':
                    spec['hold'] = 8.
                if kind == 'whip':
                    spec['stroke_length'] = 1.1
                if kind == 'random':
                    spec['seed'] = 100+int(level*10); spec['seconds'] = 15.
                if kind == 'figure8':
                    spec['plane'] = 'xy'; spec['laps'] = 1
                if kind == 'line':
                    spec['amplitude'] = .8
                if kind == 'vertical':
                    spec['amplitude'] = .4
                out.append(spec)
    return out


def _prepare(parameters):
    firmware, stack = parameters

    def one(item):
        label, config, spec = item
        return simulate_part(config, firmware, spec, stack=stack)
    return one


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--setup', default='cf_6')
    p.add_argument('--stack', type=Path, default=FLIGHT_STACK)
    p.add_argument('--kinds', default=','.join(KINDS))
    p.add_argument('--headings', default='0,90,45,-45')
    p.add_argument('--levels', default='1,2,3,4,5,6,7,8')
    p.add_argument('--members', type=int, default=12, help='sampled ensemble members (plus the adopted model and the damping brackets)')
    p.add_argument('--sigma', type=float, default=2.)
    p.add_argument('--brackets', default='damping/3; damping/10; drone slower; drone slower, damping/3',
                   help='bracket members to include, separated by ; (names of ensemble.BRACKETS), or none')
    p.add_argument('--delta', type=float, default=0., help='share of members allowed to violate a constraint (0: all must pass)')
    p.add_argument('--rise-margin', type=float, default=.04,
                   help='m the cable\'s highest node must stay below the attachment in every member: the hanging cable sits at '
                        '-0.085 (the first marker), 0 is the first link horizontal; 0.04 keeps it below ~60 deg. The simulated '
                        'closed loop under-commands sharp strokes by ~20 %% until the drone is refitted on this drone '
                        '(closed_loop_check.py), so the envelope step rule, not this margin, is the protection near the limit')
    p.add_argument('--tilt-limit', type=float, default=50.)
    p.add_argument('--error-limit', type=float, default=.5)
    p.add_argument('--motor-floor', type=float, default=.05, help='smallest motor force as a share of its hover share')
    p.add_argument('--step', type=float, default=1., help='m/s^2 above the envelope a part may go')
    p.add_argument('--first-level', type=float, default=2., help='highest level for a kind and heading never flown')
    p.add_argument('--flown-weight', type=float, default=.3)
    p.add_argument('--budget', type=float, default=80., help='seconds of motion in the plan')
    p.add_argument('--cable-fit', type=Path, default=ARCHIVE/'cable'/'20261028-012011.json')
    p.add_argument('--drone-fit', type=Path, default=ARCHIVE/'drone'/'20261027-201939.json')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--report', type=Path)
    args = p.parse_args()
    t0 = time.time()
    cable = not args.setup.endswith('_nocable')
    kinds = [k for k in args.kinds.split(',') if k]
    headings = [float(h) for h in args.headings.split(',')]
    levels = [float(x) for x in args.levels.split(',')]
    table, clean = envelope_table(DEFAULT_ROOT/'envelope'/f'{args.setup}.json')
    specs = candidates(kinds, headings, levels, clean, args.step, args.first_level, cable)
    config = fit_configuration(load_model_configuration(DEFAULT_MODEL))
    if not cable:
        config = flight_drone.drone_only(config)
    firmware = flown_gains(args.stack)
    ensemble = Ensemble(config, load_fit(args.cable_fit), load_fit(args.drone_fit) if args.drone_fit.exists() else None)
    members = ensemble.samples(args.members+1, sigma=args.sigma)
    wanted = [b.strip() for b in args.brackets.split(';') if b.strip() and b.strip() != 'none']
    members += [(name, c) for name, c in ensemble.brackets() if name in wanted]
    print(f'{len(specs)} candidate parts x {len(members)} members ({", ".join(n for n, _ in members if not n.startswith("sample"))}'
          f' + {sum(n.startswith("sample") for n, _ in members)} samples); envelope: {len(clean)} (kind, heading) flown cleanly', flush=True)
    items = [(label, c, spec) for spec in specs for label, c in members]
    runner = ProcessRunner(items, _prepare)
    results = [r for r, _ in runner((firmware, args.stack))]
    runner.close()
    report = []
    for k, spec in enumerate(specs):
        runs = results[k*len(members):(k+1)*len(members)]
        rows = []
        for (label, _), r in zip(members, runs):
            if r is None or r['failed'] or not len(r['t']):
                rows.append(dict(label=label, ok=False, reason=(r or {}).get('failed', 'no result')))
                continue
            rise, tilt, err, motor = r['rise'].max(), r['tilt'].max(), r['error'].max(), r['motor'].min()
            reasons = []
            if rise > -args.rise_margin: reasons.append(f'cable rise {rise:+.2f} m')
            if tilt > args.tilt_limit: reasons.append(f'tilt {tilt:.0f}')
            if err > args.error_limit: reasons.append(f'error {err:.2f} m')
            if motor < args.motor_floor: reasons.append(f'motor {motor:.2f}')
            rows.append(dict(label=label, ok=not reasons, reason='; '.join(reasons), rise=float(rise), tilt=float(tilt),
                             error=float(err), motor=float(motor)))
        good = [r for r in rows if r['ok']]
        feasible = len(good) >= (1.-args.delta)*len(rows)-1e-9
        # disagreement across the members that ran: spread of the tip path and the tilt over the part
        ran = [(lab, r) for (lab, _), r in zip(members, runs) if r is not None and not r['failed'] and len(r['t'])]
        if len(ran) >= 2:
            n = min(len(r['t']) for _, r in ran)
            tips = np.stack([r['tip'][:n] for _, r in ran]); tilts = np.stack([r['tilt'][:n] for _, r in ran])
            disagreement = float(np.mean(np.linalg.norm(tips.std(0), axis=1)))+.5*float(np.mean(np.radians(tilts).std(0)))
        else:
            disagreement = 0.
        key = (spec['kind'], spec.get('heading'), spec['level'])
        novelty = args.flown_weight if table.get(key, {}).get('clean', 0) > 0 else 1.
        duration = float(ran[0][1]['t'][-1]) if ran else 0.
        report.append(dict(spec=spec, feasible=bool(feasible), passed=len(good), members=len(rows), disagreement=disagreement,
                           novelty=novelty, score=disagreement*novelty if feasible else 0., duration_s=duration, rows=rows,
                           worst_rise=max((r.get('rise', -9.) for r in rows), default=None)))
    order = sorted(report, key=lambda r: -r['score'])
    chosen, used, total = [], set(), 0.
    for r in order:
        if not r['feasible'] or r['score'] <= 0:
            continue
        kh = (r['spec']['kind'], r['spec'].get('heading'))
        if kh in used or total+r['duration_s'] > args.budget:
            continue
        chosen.append(r); used.add(kh); total += r['duration_s']
    chosen.sort(key=lambda r: (r['spec']['level'], r['spec']['kind']))
    plan = dict(name=args.output.stem, setup=args.setup, parts=[r['spec'] for r in chosen],
                planned=dict(members=len(members), motion_s=round(total, 1), rise_margin_m=args.rise_margin, step=args.step))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump(plan, sort_keys=False))
    print(f'\n{"part":28s} {"feasible":>8s} {"passed":>6s} {"worst rise":>10s} {"disagree":>8s} {"novel":>5s} {"score":>6s}')
    for r in order:
        s = r['spec']; label = f"{s['kind']} {s.get('heading', '')} {s['level']:g}"
        wr = f"{r['worst_rise']:+.2f}" if r['worst_rise'] is not None else '-'
        print(f'{label:28s} {"yes" if r["feasible"] else "no":>8s} {r["passed"]:3d}/{r["members"]:<2d} {wr:>10s} {r["disagreement"]:8.3f} {r["novelty"]:5.1f} {r["score"]:6.3f}')
    print(f'\nplan ({total:.0f} s of motion, {len(chosen)} parts) -> {args.output}: ' + '; '.join(f"{r['spec']['kind']} {r['spec'].get('heading', '')} {r['spec']['level']:g}" for r in chosen))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(dict(setup=args.setup, members=[n for n, _ in members], candidates=report, plan=plan,
                                               seconds=round(time.time()-t0)), indent=1, default=float)+'\n')
    print(f'{time.time()-t0:.0f} s')


if __name__ == '__main__':
    main()
