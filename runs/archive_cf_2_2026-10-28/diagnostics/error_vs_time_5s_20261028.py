"""5 s coupled open-loop replay: error against time into the window, and tip error against measured tip speed."""
import json, sys, tomllib
from pathlib import Path
import numpy as np
ROOT = Path('/home/tony-lab/Documents/PHD/AeroWhip_deploy')
sys.path.insert(0, str(ROOT))
from aerowhip.fitting import coupled, flight_drone
from aerowhip.fitting.least_squares import ProcessRunner
from aerowhip.physics.model_configuration import DEFAULT_MODEL, load_model_configuration
from aerowhip.takes.calibration_flights import DT

biases = json.loads((ROOT/'runs/fits/coupled/20261028-015741.json').read_text())['biases_applied_m_s2']


def _prepare(parameters):
    config, firmware = parameters
    replay = flight_drone.DroneReplay(config, firmware)

    def one(item):
        w, flight = item
        out = {}
        for label, b in (('raw', None), ('bias_cancelled', biases.get(str(flight.folder)))):
            p, R, m, mv = replay.run(w, flight, pipeline=True, markers=True, attitude=True, bias=b)
            rel_sim = m-p[:, None]
            rel_real = w.markers[1:]-w.position[1:, None]
            seen = w.seen[1:]
            out[label] = dict(pos=np.linalg.norm(p-w.position[1:], axis=1),
                              tip=np.where(seen[:, -1], np.linalg.norm(m[:, -1]-w.markers[1:, -1], axis=1), np.nan),
                              shape=np.sqrt(np.nanmean(np.where(seen, np.sum((rel_sim-rel_real)**2, 2), np.nan), 1)))
        tipv = np.gradient(w.markers[:, -1], DT, axis=0)[1:]
        out['tip_speed'] = np.where(w.seen[1:, -1], np.linalg.norm(tipv, axis=1), np.nan)
        out['drone_speed'] = np.linalg.norm(np.gradient(w.position, DT, axis=0)[1:], axis=1)
        return out
    return one


def main():
    config = load_model_configuration(DEFAULT_MODEL)
    firmware = tomllib.loads(flight_drone.FIRMWARE.read_text())
    train, held = coupled.segments(5.)
    items = [it for g in held.values() for it in g]
    names = [n for n, g in held.items() for _ in g]
    runner = ProcessRunner(items, _prepare)
    res = [r for r, _ in runner((config, firmware))]
    runner.close()
    ok = [(n, r) for n, r in zip(names, res) if r is not None]
    print(f'held-out 5 s windows: {len(ok)} ok, {len(res)-len(ok)} failed; flights {sorted(set(names))}')
    n = min(len(r['raw']['pos']) for _, r in ok)
    for label in ('raw', 'bias_cancelled'):
        print(f'\n{label}: RMS over windows at time t (mm)')
        print('  t (s)  drone   shape   tip')
        for t in (0.5, 1., 2., 3., 4., 5.):
            k = min(round(t/DT)-1, n-1)
            pos = np.sqrt(np.nanmean([r[label]['pos'][k]**2 for _, r in ok]))
            sh = np.sqrt(np.nanmean([r[label]['shape'][k]**2 for _, r in ok]))
            tip = np.sqrt(np.nanmean([r[label]['tip'][k]**2 for _, r in ok]))
            print(f'  {t:4.1f}  {pos*1e3:6.0f}  {sh*1e3:6.0f}  {tip*1e3:6.0f}')
    print('\nbias_cancelled, tip error by measured tip speed (all frames, all 5 s), and drone error alongside')
    tip = np.concatenate([r['bias_cancelled']['tip'] for _, r in ok])
    sh = np.concatenate([r['bias_cancelled']['shape'] for _, r in ok])
    sp = np.concatenate([r['tip_speed'] for _, r in ok])
    for lo, hi in ((0, .5), (.5, 1.5), (1.5, 3), (3, 99)):
        s = (sp >= lo) & (sp < hi) & np.isfinite(tip)
        print(f'  tip {lo:3.1f}-{hi if hi < 99 else "":>3} m/s  frames {s.sum():6d}  tip {np.sqrt(np.mean(tip[s]**2))*1e3:5.0f} mm  shape {np.sqrt(np.nanmean(sh[s]**2))*1e3:5.0f} mm')
    print('\nper flight, bias cancelled, RMS over the whole 5 s (mm): drone / shape / tip, max tip speed')
    for name in sorted(set(n for n, _ in ok)):
        rs = [r for m, r in ok if m == name]
        f = lambda k: np.sqrt(np.nanmean(np.concatenate([r['bias_cancelled'][k] for r in rs])**2))*1e3
        print(f'  {name:28s} windows {len(rs):2d}  {f("pos"):5.0f} / {f("shape"):4.0f} / {f("tip"):4.0f}   max tip {np.nanmax(np.concatenate([r["tip_speed"] for r in rs])):.1f} m/s')


if __name__ == '__main__':
    main()
