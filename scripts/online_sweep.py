"""Robustness of the online estimator over many recorded flights, replayed as if live: does it run
on every kind of flight (with and without cable, cf_2 and cf_6, completed and crashed) without
failing, within its time budget, and without running away? Not a model-quality check.

    python scripts/online_sweep.py --roots <calibration_data>/archive/cf_2_2026-10-28 --model-commit 45900ad
    python scripts/online_sweep.py --roots <calibration_data>/cf_6 <calibration_data>/cf_6_nocable

Per flight: the run finished; updates; median and largest update time; every estimate finite and
inside its bounds; the largest distance of any parameter from the prior (in prior standard
deviations); window statuses; the reasons windows were not used. A table and runs/online/sweep_*.json.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def flights(roots):
    out = []
    for root in roots:
        for f in sorted(Path(root).rglob('flight.jsonl')):
            d = f.parent
            if (d/'poses.jsonl').exists() and (d/'parameters.json').exists():
                out.append(d)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--roots', nargs='+', type=Path, required=True)
    p.add_argument('--model-commit')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--limit', type=int)
    p.add_argument('--name', default='sweep')
    args = p.parse_args()
    folders = flights(args.roots)[:args.limit]
    out_dir = ROOT/'runs'/'online'/args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    print(f'{len(folders)} flights')
    for folder in folders:
        rel = '/'.join(folder.parts[-3:])
        out = out_dir/('_'.join(folder.parts[-3:])+'.jsonl')
        cmd = [sys.executable, str(ROOT/'scripts'/'online_fit.py'), '--flight', str(folder), '--replay', '--quiet',
               '--threads', str(args.threads), '--output', str(out)]
        if args.model_commit:
            cmd += ['--model-commit', args.model_commit]
        t0 = time.time()
        run = subprocess.run(cmd, capture_output=True, text=True)
        row = dict(flight=rel, ok=run.returncode == 0, wall_s=round(time.time()-t0, 1))
        if run.returncode != 0:
            row['error'] = (run.stderr.strip().splitlines() or ['?'])[-1][:200]
            results.append(row)
            print(f"{rel:40s} FAILED: {row['error']}")
            continue
        recs = [json.loads(line) for line in out.read_text().splitlines()]
        header = recs[0]
        updates = [r for r in recs if r['kind'] == 'update']
        names = header['parameters']
        theta = np.array([[u['theta'][n] for n in names] for u in updates]) if updates else np.zeros((0, len(names)))
        prior = np.array([header['prior'][n]['value'] for n in names]); sd = np.array([header['prior'][n]['sd'] for n in names])
        status, reasons = Counter(), Counter()
        for u in updates:
            for fam, r in u['families'].items():
                status[f"{fam} {r['status']}"] += 1
                if r['status'] not in ('accepted',):
                    reasons[r.get('reason', '').split(' (')[0]] += 1
        seconds = [u['seconds'] for u in updates if u['seconds'] > .1]
        row.update(updates=len(updates), finite=bool(np.isfinite(theta).all()),
                   median_s=float(np.median(seconds)) if seconds else 0., max_s=float(max(seconds, default=0.)),
                   max_distance_sd=float(np.abs((theta-prior)/sd).max()) if len(theta) else 0.,
                   status=dict(status), reasons=dict(reasons.most_common(6)))
        results.append(row)
        print(f"{rel:40s} {len(updates):4d} upd  {row['median_s']:.2f}/{row['max_s']:.2f} s  max {row['max_distance_sd']:4.1f} sd  "
              + ' '.join(f'{k}:{v}' for k, v in sorted(status.items())), flush=True)
    summary = dict(flights=len(results), failed=[r['flight'] for r in results if not r['ok']],
                   nonfinite=[r['flight'] for r in results if r.get('finite') is False],
                   over_budget=[r['flight'] for r in results if r.get('max_s', 0) > 1.],
                   reasons=dict(sum((Counter(r.get('reasons', {})) for r in results), Counter()).most_common(12)))
    (out_dir/'sweep.json').write_text(json.dumps(dict(summary=summary, flights=results), indent=1)+'\n')
    print('\nsummary:', json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
