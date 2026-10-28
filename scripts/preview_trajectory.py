"""Simulate one planned trajectory for display: the adopted model (configured model with the fits
used by the planner) flown closed loop at the flown gains, every cable node and the drone pose at
50 Hz. Writes <trajectory>_preview.npz beside the trajectory file, which the Identification Loop
window plays back in 3D. One simulation on one core.

    python scripts/preview_trajectory.py runs/planning/excitation_001.json
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
from aerowhip.physics.model import Model
from aerowhip.physics.model_configuration import DEFAULT_MODEL, fit_configuration, load_model_configuration
from aerowhip.planning.ensemble import Ensemble, load_fit
from aerowhip.planning.simulate import FLIGHT_STACK, simulate_motion, flight_script
from plan_flight import ARCHIVE, flown_gains  # noqa: E402

EVERY = 2          # simulation steps (10 ms) per displayed frame


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('trajectory', type=Path)
    p.add_argument('--cable-fit', type=Path, default=ARCHIVE/'cable'/'20261028-012011.json')
    p.add_argument('--drone-fit', type=Path, default=ARCHIVE/'drone'/'20261027-201939.json')
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    t0 = time.time()
    data = json.loads(args.trajectory.read_text())
    _, trajectories = flight_script()
    motion = trajectories.TrajectoryFile(data, str(data.get('name', args.trajectory.stem)))
    config = fit_configuration(load_model_configuration(DEFAULT_MODEL))
    ensemble = Ensemble(config, load_fit(args.cable_fit), load_fit(args.drone_fit) if args.drone_fit.exists() else None)
    adopted = ensemble.member_config(ensemble.cable_fit['best']['x'], ensemble.drone_best())
    r = simulate_motion(adopted, flown_gains(FLIGHT_STACK), motion, markers_every=EVERY, nodes=True)
    every = slice(None, None, EVERY)
    n = len(r['nodes'])
    model = Model(adopted)
    out = args.output or args.trajectory.with_name(args.trajectory.stem+'_preview.npz')
    np.savez_compressed(out, t=r['marker_t'][:n], position=np.asarray(r['position'])[every][:n], rotation=r['rotation'][:n],
                        nodes=r['nodes'], reference=r['reference'], rise=r['rise'][every][:n], tilt=r['tilt'][every][:n],
                        motor=r['motor'][every][:n], tip_speed=r['tip_speed'][every][:n], command=np.asarray(r['command'])[every][:n],
                        marker_indices=np.asarray(model.marker_indices), attachment=np.asarray(model.attachment_tracking),
                        failed=str(r['failed'] or ''), source=str(args.trajectory))
    print(f'{n} frames over {r["marker_t"][n-1]:.1f} s; peak cable rise {r["rise"].max():+.3f} m, peak tilt {r["tilt"].max():.0f} deg, '
          f'smallest motor {r["motor"].min():.2f}; {"FAILED: "+r["failed"] if r["failed"] else "completed"} -> {out} ({time.time()-t0:.0f} s)')


if __name__ == '__main__':
    main()
