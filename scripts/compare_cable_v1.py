"""Score the previous cable model (v1: material angles, GJ, Ct) on the same calibration windows.

Loads a saved v1 native engine binary (it cannot share a process with the current engine) and
scores it at its configured parameters and at its previous best multipliers (Cb x4, drag x4).

    python scripts/compare_cable_v1.py --engine path/to/v1/_native...so --config path/to/v1/drone_cable.json --output runs/.../cable_v1.json
"""
import argparse
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
import tomllib

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aerowhip.fitting import flight_cable
from aerowhip.fitting.calibration_set import split
from aerowhip.paths import ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location('_native', args.engine)
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    base = json.loads(args.config.read_text())
    base['controller'] = tomllib.loads((ROOT/'configs/controllers/geometric.toml').read_text())
    train, held = split('cf_2')
    results = {}
    for name, scale in (('configured', (1., 1.)), ('previous best (Cb x4, drag x4)', (4., 4.))):
        config = deepcopy(base)
        config['cable']['Cb_n_m2_s'] *= scale[0]
        config['cable']['external_drag_s_inv'] *= scale[1]
        model = native.Model(config, None)
        row = {}
        for label, groups in (('fit', train), ('held_out', held)):
            every = [w for ws in groups.values() for w in ws]
            row[label+'_rms_m'] = flight_cable.rms(flight_cable.evaluate(model, every, state_width=7))
            row[label+'_per_flight_rms_m'] = {m: flight_cable.rms(flight_cable.evaluate(model, ws, state_width=7)) for m, ws in groups.items()}
        row['parameters'] = {k: config['cable'][k] for k in ('EI_n_m2', 'Cb_n_m2_s', 'external_drag_s_inv', 'GJ_n_m2', 'Ct_n_m2_s')}
        results[name] = row
        print(name, f"fit {row['fit_rms_m']*1e3:.2f} mm, held out {row['held_out_rms_m']*1e3:.2f} mm", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(model='der_material_kelvin_voigt_v1', engine=str(args.engine), results=results), indent=1))


if __name__ == '__main__':
    main()
