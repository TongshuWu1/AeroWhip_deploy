"""Optimise the next excitation flight: a free trajectory (sum of sines per axis, at rest at both
ends) that teaches the fit the most about the parameters it is least sure of, that every ensemble
member predicts safe, and that stays within one step of the measured envelope. Writes the file the
flight script flies (`acceleration_circle_cf2.py --trajectory FILE`) and a report.

Per candidate: (1) the trajectory is normalised to the peak horizontal acceleration `--level`
(default: the highest level flown cleanly in the envelope plus --step, or --first-level);
(2) constraints: every ensemble member is flown closed loop (aerowhip.planning.simulate), the
cable's highest node must stay --rise-margin below the attachment, tilt below --tilt-limit,
position error below --error-limit, motors above --motor-floor; (3) for feasible candidates the
Fisher information about the fitted parameters (aerowhip.planning.information: one simulation per
parameter on the adopted model), scored by the weakest parameter. Random restarts, then a (1+lambda)
evolution strategy around the best, each generation evaluated in parallel.

    python scripts/optimize_trajectory.py --setup cf_6 --output runs/planning/excitation_001.json
"""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'scripts'))
from aerowhip.fitting import flight_drone
from aerowhip.fitting.calibration_set import DEFAULT_ROOT
from aerowhip.fitting.least_squares import ProcessRunner
from aerowhip.physics.model_configuration import DEFAULT_MODEL, fit_configuration, load_model_configuration
from aerowhip.planning.ensemble import Ensemble, load_fit
from aerowhip.planning.information import fisher, parameter_set, perturbed_configs, scores
from aerowhip.planning.simulate import FLIGHT_STACK, simulate_motion
from aerowhip.planning.trajectory import FourierTrajectory
from plan_flight import ARCHIVE, envelope_table, flown_gains  # noqa: E402


def _prepare(parameters):
    firmware, stack = parameters

    def one(item):
        theta_index, label, config, motion = item
        return simulate_motion(config, firmware, motion, stack=stack)
    return one


def evaluate(thetas, T, members, ensemble, params, firmware, stack, limits):
    """Constraints on every member and, where feasible, the information, for a batch of thetas."""
    motions = [T.motion(th) for th in thetas]
    adopted = ('adopted', ensemble.member_config(ensemble.cable_fit['best']['x'], ensemble.drone_best()))
    perturbed = perturbed_configs(ensemble, params)
    items = []
    for k, motion in enumerate(motions):
        for label, config in members:
            items.append((k, 'member:'+label, config, motion))
        items.append((k, 'info:adopted', adopted[1], motion))
        for name, config in perturbed:
            items.append((k, 'info:'+name, config, motion))
    runner = ProcessRunner(items, _prepare)
    results = [r for r, _ in runner((firmware, stack))]
    runner.close()
    out = []
    per = len(members)+1+len(perturbed)
    for k, th in enumerate(thetas):
        rs = results[k*per:(k+1)*per]
        member_rows = []
        for (label, _), r in zip(members, rs[:len(members)]):
            if r is None or r['failed'] or not len(r['t']):
                member_rows.append(dict(label=label, ok=False, reason=(r or {}).get('failed', 'no result')))
                continue
            rise, tilt, err, motor = float(r['rise'].max()), float(r['tilt'].max()), float(r['error'].max()), float(r['motor'].min())
            reasons = []
            if rise > -limits['rise_margin']: reasons.append(f'rise {rise:+.2f}')
            if tilt > limits['tilt']: reasons.append(f'tilt {tilt:.0f}')
            if err > limits['error']: reasons.append(f'error {err:.2f}')
            if motor < limits['motor']: reasons.append(f'motor {motor:.2f}')
            member_rows.append(dict(label=label, ok=not reasons, reason='; '.join(reasons), rise=rise, tilt=tilt, error=err, motor=motor))
        feasible = all(m['ok'] for m in member_rows)
        info = None
        base, rest = rs[len(members)], rs[len(members)+1:]
        if feasible and base is not None and not base['failed'] and all(r is not None and not r['failed'] for r in rest):
            info = scores(fisher(base, rest, params), params)
        out.append(dict(theta=np.asarray(th).tolist(), feasible=feasible, members=member_rows, information=info,
                        score=info['score'] if (feasible and info) else -1e9, summary=T.summary(th),
                        worst_rise=max((m.get('rise', -9.) for m in member_rows), default=None)))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--setup', default='cf_6')
    p.add_argument('--stack', type=Path, default=FLIGHT_STACK)
    p.add_argument('--duration', type=float, default=60.)
    p.add_argument('--terms', type=int, default=4)
    p.add_argument('--level', type=float, help='peak horizontal acceleration m/s^2 (default: envelope + step)')
    p.add_argument('--step', type=float, default=1.)
    p.add_argument('--first-level', type=float, default=2.)
    p.add_argument('--members', type=int, default=6)
    p.add_argument('--sigma', type=float, default=2.)
    p.add_argument('--brackets', default='damping/3; drone slower, damping/3')
    p.add_argument('--restarts', type=int, default=12, help='random candidates in the first generation')
    p.add_argument('--generations', type=int, default=3, help='(1+lambda) refinement rounds after the restarts')
    p.add_argument('--offspring', type=int, default=8)
    p.add_argument('--rise-margin', type=float, default=.04)
    p.add_argument('--tilt-limit', type=float, default=50.)
    p.add_argument('--error-limit', type=float, default=.5)
    p.add_argument('--motor-floor', type=float, default=.05)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--cable-fit', type=Path, default=ARCHIVE/'cable'/'20261028-012011.json')
    p.add_argument('--drone-fit', type=Path, default=ARCHIVE/'drone'/'20261027-201939.json')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--report', type=Path)
    args = p.parse_args()
    t0 = time.time()
    cable = not args.setup.endswith('_nocable')
    table, clean = envelope_table(DEFAULT_ROOT/'envelope'/f'{args.setup}.json')
    level = args.level if args.level is not None else (max(clean.values())+args.step if clean else args.first_level)
    config = fit_configuration(load_model_configuration(DEFAULT_MODEL))
    if not cable:
        config = flight_drone.drone_only(config)
    firmware = flown_gains(args.stack)
    ensemble = Ensemble(config, load_fit(args.cable_fit), load_fit(args.drone_fit) if args.drone_fit.exists() else None)
    members = ensemble.samples(args.members+1, sigma=args.sigma)
    wanted = [b.strip() for b in args.brackets.split(';') if b.strip() and b.strip() != 'none']
    members += [(n, c) for n, c in ensemble.brackets() if n in wanted]
    params = parameter_set(ensemble)
    T = FourierTrajectory(terms=args.terms, duration=args.duration, name=args.output.stem)
    limits = dict(rise_margin=args.rise_margin, tilt=args.tilt_limit, error=args.error_limit, motor=args.motor_floor)
    rng = np.random.default_rng(args.seed)
    print(f'level {level:g} m/s^2 (envelope clean max {max(clean.values()) if clean else None}); {len(members)} members; '
          f'{len(params)} parameters; {args.restarts} restarts + {args.generations} x {args.offspring} offspring; '
          f'{len(members)+1+len(params)} simulations of {args.duration:g} s per candidate', flush=True)
    history = []
    thetas = [T.normalise(T.random(rng), level) for _ in range(args.restarts)]
    best = None
    for generation in range(args.generations+1):
        rows = evaluate(thetas, T, members, ensemble, params, firmware, args.stack, limits)
        for r in rows:
            r['generation'] = generation
        history += rows
        feasible = [r for r in rows if r['feasible']]
        top = max(rows, key=lambda r: r['score'])
        if best is None or top['score'] > best['score']:
            best = top
        print(f'generation {generation}: {len(feasible)}/{len(rows)} feasible, best score {best["score"]:.3f} '
              f'(weakest {best["information"]["weakest"] if best["information"] else "-"}), {time.time()-t0:.0f} s', flush=True)
        if generation == args.generations:
            break
        lo, hi = T.bounds()
        spread = .3*(.7**generation)*(hi-lo)
        base_theta = np.asarray(best['theta'])
        thetas = [T.normalise(np.clip(base_theta+rng.normal(0., spread), lo, hi), level) for _ in range(args.offspring)]
    if not best['feasible']:
        raise SystemExit('no feasible trajectory found; lower --level or widen --error-limit')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    T.write(best['theta'], args.output, extra=dict(level=level, score=best['score'], weakest=best['information']['weakest'],
                                                   marginal_log10=best['information']['marginal_log10'], members=[n for n, _ in members],
                                                   worst_rise=best['worst_rise'], summary=best['summary']))
    print(f'\nbest: score {best["score"]:.3f}, weakest parameter {best["information"]["weakest"]}, worst member rise {best["worst_rise"]:+.2f} m, '
          f'{best["summary"]} -> {args.output}')
    for name, v in sorted(best['information']['marginal_log10'].items(), key=lambda kv: kv[1]):
        print(f'  {name:28s} {v:6.2f}')
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(dict(setup=args.setup, level=level, members=[n for n, _ in members], parameters=[p[0] for p in params],
                                               candidates=history, best=best, seconds=round(time.time()-t0)), indent=1, default=float)+'\n')
    print(f'{time.time()-t0:.0f} s')


if __name__ == '__main__':
    main()
