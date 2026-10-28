"""Validation of the online estimator on recorded flights, replayed as if live (no flight needed).

For each flight: the estimator runs from the prior (the offline fits) and from a deliberately wrong
start, and the whole flight is then scored (cable: signed marker error over 1 s windows; drone:
the drone residual over 1 s windows) at the prior, at the wrong start and at each final estimate.
Pass: from the wrong start the estimate moves back towards the prior mean (each parameter's
distance in prior standard deviations shrinks) and the flight is fitted at least as well as at the
wrong start; from the prior the flight is fitted at least as well as at the prior. Flights with
cable flips must gate those windows and not change the estimate because of them.

    python scripts/online_validation.py --model-commit 45900ad --flights <archive>/cf_2/strokes/001 ...
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'scripts'))
from aerowhip.online.estimator import ModelSpace, cable_residual, drone_residual, firmware_for  # noqa: E402
from aerowhip.planning.ensemble import load_fit  # noqa: E402
from aerowhip.takes.calibration_flights import load, windows  # noqa: E402
from online_fit import base_model  # noqa: E402
from plan_flight import ARCHIVE  # noqa: E402

WRONG = 'log10 EI=+0.3, log10 Cb=+-0.15, Cd=*1.4, motor_response_s=*1.6, body_drag_xy_s_inv=*1.4, command_delay_s=+0.012'


def score(space, theta, flight, params, threads=12):
    ws = windows(flight, duration_s=1., stride_s=1.)
    firmware = firmware_for(params)
    has_cable = flight.markers is not None

    def one(w):
        out = {}
        for name, fn in (('cable', lambda c: cable_residual(c, w)), ('drone', lambda c: drone_residual(c, firmware, w, flight))):
            if name == 'cable' and not has_cable:
                continue
            try:
                r = fn(space.config(theta, cable=has_cable))
                out[name] = r[np.isfinite(r)]
            except Exception:
                pass
        return out
    with ThreadPoolExecutor(threads) as pool:
        rows = list(pool.map(one, ws))
    return {k: float(np.sqrt(np.mean(np.concatenate([r[k] for r in rows if k in r])**2))) for k in ('cable', 'drone')
            if any(k in r for r in rows)}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--flights', nargs='+', type=Path, required=True)
    p.add_argument('--model-commit')
    p.add_argument('--wrong', default=WRONG)
    p.add_argument('--output', type=Path, default=ROOT/'runs'/'online'/'validation.json')
    args = p.parse_args()
    space = ModelSpace(base_model(args.model_commit), load_fit(ARCHIVE/'cable'/'20261028-012011.json'), load_fit(ARCHIVE/'drone'/'20261027-201939.json'))
    prior = np.array([q.value for q in space.parameters]); sd = np.array([q.sd for q in space.parameters])
    from online_fit import parse_start
    wrong = parse_start(args.wrong, space)
    results = []
    for folder in args.flights:
        name = f'{folder.parent.name}/{folder.name}'
        runs = {}
        for label, start in (('prior', None), ('wrong', args.wrong)):
            out = ROOT/'runs'/'online'/f'validation_{folder.parent.name}_{folder.name}_{label}.jsonl'
            cmd = [sys.executable, str(ROOT/'scripts'/'online_fit.py'), '--flight', str(folder), '--replay', '--quiet', '--output', str(out)]
            if args.model_commit:
                cmd += ['--model-commit', args.model_commit]
            if start:
                cmd += ['--start', start]
            subprocess.run(cmd, check=True, capture_output=True)
            rows = [json.loads(line) for line in out.read_text().splitlines()]
            summary = [r for r in rows if r['kind'] == 'summary'][0]
            updates = [r for r in rows if r['kind'] == 'update']
            runs[label] = dict(theta=np.array([summary['theta'][n] for n in space.names]), counts=summary['counts'],
                               seconds=float(np.median([u['seconds'] for u in updates if u['seconds'] > .1] or [0])),
                               gated={f: sorted({u['families'][f].get('reason', '') for u in updates if u['families'].get(f, {}).get('status') == 'gated'}) for f in summary['counts']})
        flight = load(folder)
        params = json.loads((folder/'parameters.json').read_text())
        s = {k: score(space, th, flight, params) for k, th in (('prior', prior), ('wrong start', wrong),
                                                                ('final from prior', runs['prior']['theta']), ('final from wrong', runs['wrong']['theta']))}
        before = np.abs(wrong-prior)/sd
        after = np.abs(runs['wrong']['theta']-prior)/sd
        print(f'\n{name}')
        print('  ' + '  '.join(f'{k}: ' + ', '.join(f'{f} {v*1000 if f == "cable" else v:.2f}' for f, v in d.items()) for k, d in s.items())
              + '   (cable mm per component; drone normalised)')
        print('  distance from the prior mean (sd) wrong start -> final: ' + ', '.join(f'{n.split(" ")[-1]} {b:.1f}->{a:.1f}' for n, b, a in zip(space.names, before, after)))
        print(f"  windows from the prior: {runs['prior']['counts']}; update {runs['prior']['seconds']:.2f} s")
        results.append(dict(flight=name, scores=s, start_distance_sd=before.tolist(), final_distance_sd=after.tolist(),
                            final_from_prior=runs['prior']['theta'].tolist(), final_from_wrong=runs['wrong']['theta'].tolist(),
                            counts=runs['prior']['counts'], gated=runs['prior']['gated'], update_s=runs['prior']['seconds']))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(parameters=space.names, prior=prior.tolist(), sd=sd.tolist(), wrong=wrong.tolist(), flights=results), indent=1)+'\n')
    print('->', args.output)


if __name__ == '__main__':
    main()
