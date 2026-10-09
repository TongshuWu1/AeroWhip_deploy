"""Run the compact current-pipeline regressions; never train a policy."""
import argparse
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import sys
import unittest
from project_paths import project_path

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='Optional new JSON evidence file (not overwritten)')
    args = parser.parse_args()
    if args.output:
        args.output=project_path(args.output)
    if args.output and args.output.exists():
        parser.error('Output already exists; use a new path to preserve earlier results.')
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    import torch
    from evaluation.records import active_workers
    workers = active_workers()
    if workers:
        raise RuntimeError(f'Wait for active training/evaluation workers before validation: {workers}')
    for folder in ('policy_tests', 'context_evaluation'):
        if project_path(ROOT/'runs'/folder/'worker.lock').exists():
            raise RuntimeError('Inspect the evaluation lock in runs/'+folder+' before validation; no lock was removed.')
    torch.set_num_threads(2)
    sys.path.insert(0, str(ROOT/'tests'))
    suite = unittest.defaultTestLoader.discover(str(ROOT/'tests'), pattern='test_*.py')
    log = io.StringIO()
    result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
    report = dict(
        utc=datetime.now(timezone.utc).isoformat(),
        scope='Synthetic contracts and short native numerical checks; no policy training or campaign evaluation',
        cuda_available=torch.cuda.is_available(), tests_run=result.testsRun,
        passed=result.testsRun-len(result.failures)-len(result.errors)-len(result.skipped),
        failures=[dict(test=t.id(), traceback=trace) for t, trace in result.failures],
        errors=[dict(test=t.id(), traceback=trace) for t, trace in result.errors],
        skipped=[dict(test=t.id(), reason=reason) for t, reason in result.skipped],
        log=log.getvalue())
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(log.getvalue())
    print(f"{report['passed']}/{report['tests_run']} passed; {len(result.skipped)} skipped")
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    raise SystemExit(main())
