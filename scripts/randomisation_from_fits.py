"""Training randomisation ranges from the adopted model fits.

Reads the latest adoption of the drone and cable stages (runs/fits/adoptions.jsonl), their result
files and the calibration manifest, and writes `environment.randomisation` of an experiment file:

    motor response      adopted value ± 1.96 sd (clustered by flight), at least 2 ms
    command delay       the delays of the profile whose loss is within --delay-tolerance (1 %) of
                        its minimum (the delay is profiled on a grid, it has no sd)
    body drag xy, z     scales 1 ± 1.96 sd / value of each rate
    bending damping Cb  scales 10^(± 1.96 sd) of log10 Cb
    battery voltage     lowest end-of-flight to highest start-of-flight voltage of the flights used
    acceleration bias   per axis, the RMS about zero of the per-flight biases of the drone fit (the
                        sign of the y bias changed between sessions, so the spread is centred on zero)

The model-group count is kept. Prints every number with its source; nothing else is changed.

    python scripts/randomisation_from_fits.py [--experiment configs/experiments/strike_x_v1.json]
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aerowhip.fitting.calibration_set import manifest
from aerowhip.fitting.stages import FITS
from aerowhip.paths import ROOT


def latest_adoptions():
    rows = [json.loads(line) for line in (FITS/'adoptions.jsonl').read_text().splitlines() if line.strip()]
    latest = {}
    for row in rows:
        latest[row['stage']] = row
    for stage in ('drone', 'cable'):
        if stage not in latest:
            raise SystemExit(f'No adopted {stage} result')
    return latest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=Path, default=ROOT/'configs/experiments/strike_x_v1.json')
    parser.add_argument('--delay-tolerance', type=float, default=.01, help='relative loss above the profile minimum')
    args = parser.parse_args()
    adopted = latest_adoptions()
    drone = json.loads((ROOT/adopted['drone']['result']).read_text())
    cable = json.loads((ROOT/adopted['cable']['result']).read_text())
    if drone.get('method') != 'least_squares' or cable.get('method') != 'least_squares':
        raise SystemExit('The adopted results must come from the least-squares segment fits (they carry the intervals)')
    sd = dict(zip(drone['uncertainty']['parameters'], drone['uncertainty']['standard_deviation']))
    a = drone['adopt']
    response = a['motor_response_s']
    response_range = [max(.002, response-1.96*sd['motor_response_s']), response+1.96*sd['motor_response_s']]
    drag_xy, drag_z = a['body_drag_s_inv'][0], a['body_drag_s_inv'][2]
    xy_scale = [1-1.96*sd['body_drag_xy_s_inv']/drag_xy, 1+1.96*sd['body_drag_xy_s_inv']/drag_xy]
    z_scale = [1-1.96*sd['body_drag_z_s_inv']/drag_z, 1+1.96*sd['body_drag_z_s_inv']/drag_z]
    if min(xy_scale[0], z_scale[0]) <= 0:
        raise SystemExit('A drag interval reaches zero; the log-uniform scale needs a positive lower bound')
    key = 'loss' if 'loss' in drone['delay_profile'][0] else 'rms_m'
    best = min(p[key] for p in drone['delay_profile'])
    near = [p['command_delay_s'] for p in drone['delay_profile'] if p[key] <= best*(1+args.delay_tolerance)]
    delay_range = [min(near), max(near)]
    cable_sd = dict(zip(cable['uncertainty']['parameters'], cable['uncertainty']['standard_deviation']))
    cb_scale = [10**(-1.96*cable_sd['log10 Cb']), 10**(1.96*cable_sd['log10 Cb'])]
    biases = np.array(list(drone['best']['bias_m_s2'].values()))
    bias_sd = np.sqrt(np.mean(biases**2, axis=0))
    m = manifest()
    batteries = [f['battery_start_end_v'] for f in m.data['flights']
                 if f['role'] in ('fit', 'held_out') and len(f.get('battery_start_end_v') or []) == 2]
    voltage = [min(b[1] for b in batteries), max(b[0] for b in batteries)]
    experiment = json.loads(args.experiment.read_text())
    previous = experiment['environment']['randomisation']
    ranges = dict(groups=previous['groups'], battery_voltage_v=[round(v, 3) for v in voltage],
                  acceleration_bias_sd_mps2=[round(float(v), 4) for v in bias_sd],
                  body_drag_xy_scale=[round(v, 4) for v in xy_scale], body_drag_z_scale=[round(v, 4) for v in z_scale],
                  bending_damping_scale=[round(v, 4) for v in cb_scale],
                  motor_response_s=[round(v, 5) for v in response_range], command_delay_s=[round(v, 4) for v in delay_range])
    experiment['environment']['randomisation'] = ranges
    args.experiment.write_text(json.dumps(experiment, indent=2)+'\n')
    print(f"drone fit {adopted['drone']['result']} (adopted {adopted['drone']['time']}), "
          f"cable fit {adopted['cable']['result']} (adopted {adopted['cable']['time']}); {len(batteries)} flights' batteries")
    print(f"  motor response {response*1e3:.1f} ± {1.96*sd['motor_response_s']*1e3:.1f} ms -> {ranges['motor_response_s']} s")
    print(f"  command delay: {key} within {args.delay_tolerance:.0%} of the minimum at {near} s -> {ranges['command_delay_s']} s")
    print(f"  drag xy {drag_xy:.3f} ± {1.96*sd['body_drag_xy_s_inv']:.3f}, z {drag_z:.3f} ± {1.96*sd['body_drag_z_s_inv']:.3f} 1/s "
          f"-> scales {ranges['body_drag_xy_scale']}, {ranges['body_drag_z_scale']}")
    print(f"  Cb: log10 ± {1.96*cable_sd['log10 Cb']:.3f} -> scale {ranges['bending_damping_scale']}")
    print(f"  battery {ranges['battery_voltage_v']} V; bias RMS per axis {ranges['acceleration_bias_sd_mps2']} m/s² "
          f"(mean {np.round(biases.mean(0), 3).tolist()}, {len(biases)} flights)")
    print(f'written to {args.experiment}')


if __name__ == '__main__':
    main()
