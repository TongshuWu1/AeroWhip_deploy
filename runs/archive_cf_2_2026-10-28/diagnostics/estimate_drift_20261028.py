"""How much of the drone's open-loop drift over 5 s does the onboard attitude-estimate error explain?

Drone-only flights with attitude.jsonl. Replay the sent commands open loop (adopted drone model,
no cable). The real drone points its thrust along its *estimated* body z, the true body z differs
by the estimate error, so the real drone gets an extra acceleration T (z_true - z_est) the model
does not have. Double-integrate it from the window start and see whether it cancels the replay's
position error."""
import json, sys, tomllib
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
ROOT = Path('/home/tony-lab/Documents/PHD/AeroWhip_deploy')
sys.path.insert(0, str(ROOT))
from aerowhip.fitting import flight_drone
from aerowhip.fitting.calibration_set import flight_windows
from aerowhip.physics.model_configuration import DEFAULT_MODEL, load_model_configuration
from aerowhip.takes.calibration_flights import DT, load

DATA = Path('/home/tony-lab/Documents/PHD/Crazyflie/crazyswarm2_jazzy_whipping/calibration_data')
FLIGHTS = ['cf_2_nocable/random_high/003', 'cf_2_nocable/random_high/004', 'cf_2_nocable/random_high/009',
           'cf_2_nocable/random_high/010', 'cf_2_nocable/line_star_xy/002', 'cf_2_nocable/line_star_xy/003']
G = 9.80665
E3 = np.array([0., 0., 1.])


def estimate(folder):
    rows = [json.loads(l) for l in (DATA/folder/'attitude.jsonl').read_text().splitlines() if l.strip()]
    return np.array([r['stamp'] for r in rows]), np.array([r['rpy_deg'] for r in rows])


def est_rotation(rpy, sign):
    r, p, y = np.radians(rpy).T
    return Rotation.from_euler('ZYX', np.c_[y, sign*p, r]).as_matrix()


def integrate(a):
    v = np.cumsum(a, 0)*DT
    return np.cumsum(v, 0)*DT


def main():
    config = flight_drone.drone_only(load_model_configuration(DEFAULT_MODEL))
    firmware = tomllib.loads(flight_drone.FIRMWARE.read_text())
    replay = flight_drone.DroneReplay(config, firmware)
    rows = []
    for name in FLIGHTS:
        folder = DATA/name
        flight = load(folder)
        flight.markers = None
        ts, rpy = estimate(name)
        ws = flight_windows(folder, cable=False, duration_s=5., stride_s=5.)
        # pitch sign convention: the one that matches OptiTrack's tilt best over the flight
        Rall = Rotation.from_quat(flight.quaternion).as_matrix()
        rpy_f = np.array([np.interp(flight.stamp, ts, rpy[:, k]) for k in range(3)]).T
        sign = min((1, -1), key=lambda s: np.mean(np.sum((est_rotation(rpy_f, s)@E3-Rall@E3)**2, 1)))
        z_err = Rall@E3-est_rotation(rpy_f, sign)@E3
        offset = z_err.mean(0)                     # constant part (IMU mounting, trims): the per-flight bias
        for w in ws:
            stamps = w.start_stamp+w.time[1:]
            if stamps[0] < ts[0] or stamps[-1] > ts[-1]:
                continue
            try:
                p = replay.run(w, flight, pipeline=True)
            except Exception:
                continue
            e = p-w.position[1:]
            est = np.array([np.interp(stamps, ts, rpy[:, k]) for k in range(3)]).T
            dz = w.rotation[1:]@E3-est_rotation(est, sign)@E3-offset
            thrust = np.linalg.norm(w.sent[1:]+G*E3, axis=1)
            drift = integrate(thrust[:, None]*dz)            # real = sim + drift
            rows.append(dict(name=name, e=e, drift=drift, eb=flight_drone.debias(e), db=flight_drone.debias(drift),
                             tilt_err=np.degrees(np.linalg.norm(dz[:, :2], axis=1))))
        print(f'{name}: pitch sign {sign:+d}, constant tilt offset {np.degrees(np.linalg.norm(offset[:2])):.2f} deg, windows {sum(r["name"] == name for r in rows)}', flush=True)
    h = lambda x: np.sqrt(np.mean(np.sum(x[:, :2]**2, 1)))  # horizontal RMS
    print(f'\n{len(rows)} windows of 5 s. Horizontal position error, RMS over windows (mm):')
    print('  t (s)   replay   replay+estimate error   | per-window constant accel removed: replay   +estimate')
    for t in (1., 2., 3., 4., 5.):
        k = round(t/DT)-1
        f = lambda key: np.sqrt(np.mean([np.sum(r[key][k, :2]**2) for r in rows]))*1e3
        g = lambda a, b: np.sqrt(np.mean([np.sum((r[a][k, :2]+r[b][k, :2])**2) for r in rows]))*1e3
        print(f'  {t:4.1f}   {f("e"):6.0f}   {g("e", "drift"):6.0f}                  |   {f("eb"):6.0f}   {g("eb", "db"):6.0f}')
    tilt = np.concatenate([r['tilt_err'] for r in rows])
    print(f'\nestimate tilt error (constant removed): RMS {np.sqrt(np.mean(tilt**2)):.2f} deg')
    E = np.concatenate([r['e'][:, :2] for r in rows]); D = np.concatenate([r['drift'][:, :2] for r in rows])
    print(f'correlation of replay error with -drift: x {np.corrcoef(E[:, 0], -D[:, 0])[0, 1]:.2f}, y {np.corrcoef(E[:, 1], -D[:, 1])[0, 1]:.2f}')


if __name__ == '__main__':
    main()
