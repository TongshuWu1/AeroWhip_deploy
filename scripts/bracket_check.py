"""Does the model ensemble bracket the cable's measured behaviour on a recorded stroke?

The stroke (a phase of a recorded cable flight, e.g. the sharp stop that flipped the cable) is
replayed from the sent commands (full coupled model, the flown gains) with the adopted model, the
bracket members (ensemble.BRACKETS) and `--samples` members drawn from the fits' uncertainties.
Per member: the cable's highest rise above the attachment and the drone's peak tilt, against the
raw cloud's rise. The ensemble brackets the stroke when at least one sampled member and the
adopted model enclose the measured rise, or a bracket member reaches it.

    python scripts/bracket_check.py --flight cf_6/mix_a/002 --phase "decay 0 6" --seconds 2.5
"""
import argparse
import json
from pathlib import Path
import sys
import tomllib

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from aerowhip.fitting import flight_drone
from aerowhip.fitting.calibration_set import DEFAULT_ROOT
from aerowhip.fitting.least_squares import ProcessRunner
from aerowhip.physics.model_configuration import DEFAULT_MODEL, load_model_configuration
from aerowhip.planning.ensemble import Ensemble, load_fit
from aerowhip.planning.measures import cable_rise
from aerowhip.planning.windows import phase_window, raw_cloud_rise
from aerowhip.takes.calibration_flights import load

ARCHIVE = ROOT/'runs'/'archive_cf_2_2026-10-28'/'fits'
CABLE_FIT = ARCHIVE/'cable'/'20261028-012011.json'
DRONE_FIT = ARCHIVE/'drone'/'20261027-201939.json'


def _replay(parameters):
    window, flight, firmware = parameters

    def one(item):
        label, config = item
        replay = flight_drone.DroneReplay(config, firmware)
        p, R, markers, _ = replay.run(window, flight, pipeline=True, markers=True, attitude=True)
        attachment = np.asarray(replay.model.attachment_tracking)
        root = p+np.einsum('nij,j->ni', R, attachment)
        rise = cable_rise(np.concatenate([root[:, None], markers], 1), root)
        tilt = np.degrees(np.arccos(np.clip(R[:, 2, 2], -1, 1)))
        return dict(label=label, max_rise=float(rise.max()), max_tilt=float(tilt.max()))
    return one


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--flight', default='cf_6/mix_a/002')
    p.add_argument('--phase', default='decay 0 6')
    p.add_argument('--seconds', type=float, default=2.5)
    p.add_argument('--samples', type=int, default=16)
    p.add_argument('--sigma', type=float, default=2., help='standard deviations of the fit uncertainties to sample at')
    p.add_argument('--cable-fit', type=Path, default=CABLE_FIT)
    p.add_argument('--drone-fit', type=Path, default=DRONE_FIT)
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    folder = DEFAULT_ROOT/args.flight
    flight = load(folder)
    window = phase_window(flight, args.phase, args.seconds)
    if window is None:
        raise SystemExit(f'{args.flight}: no labelled window at phase {args.phase!r}')
    config = load_model_configuration(DEFAULT_MODEL)
    firmware = tomllib.loads(flight_drone.FIRMWARE.read_text())
    for key, value in json.loads((folder/'parameters.json').read_text()).get('attitude', {}).items():
        firmware['ctrlMel'][key.split('.')[1]] = float(value)
    ensemble = Ensemble(config, load_fit(args.cable_fit), load_fit(args.drone_fit) if args.drone_fit.exists() else None)
    attachment = np.asarray(config['attachment']['position_tracking_m'])
    measured = raw_cloud_rise(folder, window, attachment)
    members = ensemble.brackets()+ensemble.samples(args.samples+1, sigma=args.sigma)[1:]
    runner = ProcessRunner(members, _replay)
    results = [r for r, _ in runner((window, flight, firmware))]
    runner.close()
    tilt_measured = np.degrees(np.arccos(np.clip(window.rotation[1:, 2, 2], -1, 1))).max()
    print(f'{args.flight} {args.phase!r}, {args.seconds:g} s: measured rise {np.nanmax(measured):+.2f} m, tilt {tilt_measured:.0f} deg')
    print(f'{"member":24s} {"rise":>8s} {"tilt":>6s}')
    rises = {}
    for r in results:
        if r is None:
            continue
        rises[r['label']] = r['max_rise']
        print(f'{r["label"]:24s} {r["max_rise"]:+8.2f} {r["max_tilt"]:6.0f}')
    target = float(np.nanmax(measured))
    reached = [k for k, v in rises.items() if v >= target]
    sampled = [v for k, v in rises.items() if k.startswith('sample')]
    enclosed = bool(sampled) and min(sampled) <= target <= max(sampled)
    print(f'\nmeasured {target:+.2f} m: reached by {reached or "no member"}; sampled members span '
          f'{min(sampled):+.2f}..{max(sampled):+.2f} m -> {"BRACKETED" if enclosed or reached else "NOT bracketed"}')
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(dict(flight=args.flight, phase=args.phase, seconds=args.seconds, measured_rise_m=target,
                                               measured_tilt_deg=float(tilt_measured), members=results, bracketed=bool(enclosed or reached)),
                                          indent=1)+'\n')


if __name__ == '__main__':
    main()
