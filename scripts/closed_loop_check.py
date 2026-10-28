"""Does the simulated closed loop command what the real one commanded? A recorded part is flown
closed loop in the simulator (configured model, flown gains, the flight script's host PID on the
same planned reference) and compared with the recording: the peak horizontal command the host
loop sent, the drone's peak tilt and the position error against the reference. The planner's
safety rests on this: on 2026-10-28 the real host loop sent 8.8 m/s^2 where 6 was planned.

    python scripts/closed_loop_check.py --flight cf_6/mix_a/002 --phase "decay 0 6"
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from aerowhip.fitting.calibration_set import DEFAULT_ROOT
from aerowhip.physics.model_configuration import DEFAULT_MODEL, fit_configuration, load_model_configuration
from aerowhip.planning.simulate import simulate_part
sys.path.insert(0, str(ROOT/'scripts'))
from plan_flight import flown_gains  # noqa: E402
from aerowhip.planning.simulate import FLIGHT_STACK  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--flight', default='cf_6/mix_a/002')
    p.add_argument('--phase', default='decay 0 6')
    p.add_argument('--hold', type=float, default=5.)
    args = p.parse_args()
    folder = DEFAULT_ROOT/args.flight
    rows = [json.loads(l) for l in (folder/'flight.jsonl').read_text().splitlines() if l.strip()]
    part = [r for r in rows if r['phase'] == args.phase]
    kind, heading, level = args.phase.split()[0], float(args.phase.split()[1]), float(args.phase.split()[2])
    real_cmd = np.array([r['sent_acceleration'][:2] for r in part]); real_plan = np.array([r['desired_acceleration'][:2] for r in part])
    real_tilt = np.array([r['tilt_deg'] for r in part])
    real_err = np.array([np.linalg.norm(np.subtract(r['desired_position'], r['position'])) for r in part])
    spec = dict(kind=kind, heading=heading, level=level, hold=args.hold)
    config = fit_configuration(load_model_configuration(DEFAULT_MODEL))
    sim = simulate_part(config, flown_gains(FLIGHT_STACK), spec)
    inside = [i for i, ph in enumerate(sim['phase']) if ph.startswith(kind)]
    cmd = np.linalg.norm(sim['command'][inside][:, :2], axis=1)
    print(f'{args.flight} {args.phase!r}: planned peak {np.linalg.norm(real_plan, axis=1).max():.1f} m/s^2')
    print(f'{"":22s} {"real":>8s} {"simulated":>10s}')
    print(f'{"sent command peak":22s} {np.linalg.norm(real_cmd, axis=1).max():8.1f} {cmd.max():10.1f}  m/s^2')
    print(f'{"drone tilt peak":22s} {real_tilt.max():8.0f} {sim["tilt"][inside].max():10.0f}  deg')
    print(f'{"position error peak":22s} {real_err.max()*1e3:8.0f} {sim["error"][inside].max()*1e3:10.0f}  mm')
    print(f'{"cable rise max":22s} {"(cloud +0.29)":>8s} {sim["rise"][inside].max():+10.2f}  m')


if __name__ == '__main__':
    main()
