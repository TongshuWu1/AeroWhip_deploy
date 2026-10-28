"""Closed-loop behaviour of the host PID on recorded flights: tracking error, and the integral term
(the slow disturbance the loop had to cancel), reconstructed from the logged command."""
import json, sys
from pathlib import Path
import numpy as np
DATA = Path('/home/tony-lab/Documents/PHD/Crazyflie/crazyswarm2_jazzy_whipping/calibration_data')
KP, KD = 4., 3.

for setup in ('cf_2_nocable', 'cf_2'):
    print(f'== {setup}')
    print(f'  {"flight":24s} {"track xy":>9s} {"track z":>8s} {"peak xy":>8s} {"I xy mean":>16s} {"I xy wander":>11s} {"I z wander":>10s} {"I tau":>6s} {"sat":>5s}')
    for f in sorted((DATA/setup).glob('*/*/flight.jsonl')):
        rows = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
        if len(rows) < 500:
            continue
        g = lambda k: np.array([r[k] for r in rows], float)
        p, v, pd, vd, ad, sent, trim = g('position'), g('velocity'), g('desired_position'), g('desired_velocity'), g('desired_acceleration'), g('sent_acceleration'), g('trim')
        sat = np.array([r['saturated'] for r in rows])
        e = pd-p
        integral = sent-ad-trim-KP*e-KD*(vd-v)          # = KI * integral where not saturated
        ok = ~sat
        I = integral[ok]
        ixy = I[:, :2]-I[:, :2].mean(0)
        # correlation time of the wander: lag where autocorrelation of x falls to 1/e (50 Hz)
        x = ixy[:, 0]-ixy[:, 0].mean()
        ac = np.correlate(x, x, 'full')[len(x)-1:]
        ac /= ac[0]
        tau = np.argmax(ac < 1/np.e)/50.
        print(f'  {f.parent.parent.name+"/"+f.parent.name:24s} {np.sqrt(np.mean(np.sum(e[:, :2]**2, 1)))*1e3:7.0f}mm {np.sqrt(np.mean(e[:, 2]**2))*1e3:6.0f}mm '
              f'{np.max(np.linalg.norm(e[:, :2], axis=1))*1e3:6.0f}mm  {np.round(I[:, :2].mean(0), 2).tolist()!s:>16s} {np.sqrt(np.mean(np.sum(ixy**2, 1))):9.3f}   '
              f'{np.std(I[:, 2]):8.3f}   {tau:5.1f}s {sat.mean()*100:4.0f}%')
