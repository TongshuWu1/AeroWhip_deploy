"""The model identification pipeline as a list of stages, for the Model Fit window and scripts.

Each stage names its script and arguments, the data it uses, the model values it fixes (read from
the configuration files), how to summarise a result file (headline before/after, per-parameter
and per-flight tables) and how to adopt a result into the configuration files. Adopting writes
only the stage's own keys and appends a provenance line to `runs/fits/adoptions.jsonl`.
"""
from dataclasses import dataclass, field
from datetime import datetime
import json
import math
from pathlib import Path
import re
import tomllib

from aerowhip.paths import ROOT

PHYSICS = ROOT/'configs/physics/drone_cable.json'
ACTUATOR = ROOT/'configs/actuators/drone.json'
CONTROLLER = ROOT/'configs/controllers/onboard.toml'
FIRMWARE = ROOT/'configs/controllers/crazyflie_firmware.toml'
FITS = ROOT/'runs/fits'
PYTHON = ROOT/'.venv/bin/python'


def _json(path):
    return json.loads(Path(path).read_text())


def _write_json(path, data):
    """Rewrite a JSON config keeping its indentation and line endings."""
    text = Path(path).read_bytes().decode('utf-8')
    newline = '\r\n' if '\r\n' in text else '\n'
    out = json.dumps(data, indent=2).replace('\n', newline)+newline
    Path(path).write_bytes(out.encode('utf-8'))


def _write_toml_value(path, key, value):
    """Replace `key = value` in a TOML file, keeping the comment after it."""
    text = Path(path).read_text()
    pattern = re.compile(rf'^({re.escape(key)}\s*=\s*)([^#\n]*?)(\s*(#.*)?)$', re.MULTILINE)
    if not pattern.search(text):
        raise KeyError(f'{key} not found in {path}')
    text = pattern.sub(lambda m: f'{m.group(1)}{value!r}{m.group(3)}', text, count=1)
    Path(path).write_text(text)


def _strong(u, threshold=.9):
    names, c = u['parameters'], u['correlation']
    return [(names[i], names[j], c[i][j]) for i in range(len(names)) for j in range(i+1, len(names)) if abs(c[i][j]) >= threshold]


def _mm(v):
    return None if v is None or (isinstance(v, float) and not math.isfinite(v)) else v*1e3


@dataclass
class Stage:
    key: str
    title: str
    description: str
    script: str
    arguments: list
    data: str
    unit: str
    adoptable: bool = True
    parameters: list = field(default_factory=list)
    writes = ()             # configuration files that adopting changes

    def output(self):
        return FITS/self.key/f'{datetime.now():%Y%m%d-%H%M%S}.json'

    def command(self, output):
        return [str(PYTHON), str(ROOT/self.script), *self.arguments, '--output', str(output)]

    def runs(self):
        """Result files of this stage, newest first (the window's run list)."""
        found = sorted((FITS/self.key).glob('*.json'), reverse=True) if (FITS/self.key).exists() else []
        return [p for p in found if not p.name.endswith('.progress.json')]

    # overridden per stage
    def current(self):
        return {}

    def summary(self, result):
        return dict(headline=[], parameters=[], flights=dict(columns=[], rows=[]))

    def adopt(self, result):
        raise NotImplementedError


def record_adoption(stage, result_path, values):
    FITS.mkdir(parents=True, exist_ok=True)
    with (FITS/'adoptions.jsonl').open('a') as f:
        f.write(json.dumps(dict(time=datetime.now().isoformat(timespec='seconds'), stage=stage,
                                result=str(result_path), values=values))+'\n')


class ThrustMap(Stage):
    writes = (FIRMWARE,)

    def current(self):
        fw = tomllib.loads(FIRMWARE.read_text())
        return {'collective thrust scale κ': fw['thrust_scale'], 'exponent p': fw['thrust_exponent'],
                'voltage slope β (1/V)': fw['voltage_slope_v_inv']}

    def summary(self, r):
        before, after = r['current'], r['best']
        adopt = r['adopt']
        names = list(self.current())
        fitted = [adopt['thrust_scale'], adopt['thrust_exponent'], adopt['voltage_slope_v_inv']]
        spread = r['motor_spread_factor']
        rows = []
        for i, (band, value) in enumerate(after['bands'].items()):
            rows.append([band, value['samples'], before['bands'].get(band, {}).get('mean_ratio'), value['mean_ratio']])
        return dict(
            headline=[('map / required thrust, rms over PWM bins (%)', before['bin_rms']*100, after['bin_rms']*100)],
            parameters=[(n, b, a) for n, b, a in zip(names, before['x'], fitted)],
            flights=dict(columns=['thrust band', 'samples', 'ratio before', 'ratio after'], rows=rows),
            notes=[f"{r['samples']} motor samples from {len(r['flights'])} flights; motor-spread factor {spread:.3f} folded into κ"])

    def adopt(self, r):
        values = r['adopt']
        for key in ('thrust_scale', 'thrust_exponent', 'voltage_slope_v_inv'):
            _write_toml_value(FIRMWARE, key, round(values[key], 6))
        return values


class Drone(Stage):
    writes = (PHYSICS, ACTUATOR, CONTROLLER)

    def current(self):
        physics, actuator = _json(PHYSICS), _json(ACTUATOR)
        controller = tomllib.loads(CONTROLLER.read_text())
        drag = physics['drone']['body_drag_s_inv']
        return {'roll/pitch inertia (kg m²)': physics['drone']['inertia_body_about_com_kg_m2'][0][0],
                'motor arm (m)': abs(actuator['motor_positions_com_m'][0][0])*2**.5,
                'motor response (s)': actuator['response_time_s'][0], 'command delay (s)': controller['command_delay_s'],
                'drag xy (1/s)': drag[0], 'drag z (1/s)': drag[2]}

    def summary(self, r):
        """Least squares on short segments (scripts/fit_drone_segments.py)."""
        return self._segments_summary(r)

    def _segments_summary(self, r):
        """Least squares on short segments (scripts/fit_drone_segments.py)."""
        start, best, adopt, u = r['start'], r['best'], r['adopt'], r['uncertainty']
        sd = dict(zip(u['parameters'], u['standard_deviation']))
        inertia_now = adopt['inertia_xy_kg_m2']/best['x'][0]
        x0 = start['x']
        before = [inertia_now*x0[0], self.current()['motor arm (m)'], x0[1], start['command_delay_s'], x0[2], x0[3]]
        after = [adopt['inertia_xy_kg_m2'], self.current()['motor arm (m)'], adopt['motor_response_s'],
                 adopt['command_delay_s'], *adopt['body_drag_s_inv'][1:]]
        ci = [f"± {1.96*sd['inertia_scale_xy']*inertia_now:.2g}", 'measured', f"± {1.96*sd['motor_response_s']:.2g}",
              'profiled', f"± {1.96*sd['body_drag_xy_s_inv']:.2g}", f"± {1.96*sd['body_drag_z_s_inv']:.2g}"]
        rows = [[name, _mm(start['per_flight_rms_m'].get(name)), _mm(value)] for name, value in best['per_flight_rms_m'].items()]
        headline = [('position error, 1 s segments (mm)', _mm(start['rms_m']), _mm(best['rms_m']))]
        if 'tilt_deg' in start:
            headline += [('tilt error (deg)', start['tilt_deg'], best['tilt_deg']),
                         ('combined loss (1 at the start)', start['loss'], best['loss'])]
        if best.get('held_out'):
            headline.append(('held out (mm)', None, _mm(best['held_out']['rms_m'])))
        profile = ', '.join(f"{p['command_delay_s']*1e3:.0f} ms {p['rms_m']*1e3:.2f}"+(f" / {p['tilt_deg']:.2f}°" if 'tilt_deg' in p else '')
                            for p in r['delay_profile'])
        strong = [f'{a} / {b} {c:+.2f}' for a, b, c in _strong(u)]
        return dict(
            headline=headline, parameters=[(n, b, a, c) for n, b, a, c in zip(self.current(), before, after, ci)],
            flights=dict(columns=['flight', 'error before', 'error after'], rows=rows),
            notes=[f'Loss: {r["loss"]}.', f'Delay profile (position mm / tilt deg): {profile}.',
                   'Intervals: 95 %, clustered by flight, linearised.' + (f' Strongly correlated: {"; ".join(strong)}.' if strong else ''),
                   f"Failed segment replays during the search: {sum(p['failed_segment_replays'] for p in r['delay_profile'])}; "
                   f"reproduced twice: {best['reproduced_twice']}"])

    def adopt(self, r):
        values = r['adopt']
        if abs(values['arm_scale']-1) > 1e-9:
            raise ValueError('this result refits the motor arm; only fits with the measured arm (--fix-arm) are adopted')
        physics, actuator = _json(PHYSICS), _json(ACTUATOR)
        inertia = physics['drone']['inertia_body_about_com_kg_m2']
        inertia[0][0] = inertia[1][1] = values['inertia_xy_kg_m2']
        physics['drone']['body_drag_s_inv'] = values['body_drag_s_inv']
        _write_json(PHYSICS, physics)
        actuator['response_time_s'] = [values['motor_response_s']]*4
        _write_json(ACTUATOR, actuator)
        _write_toml_value(CONTROLLER, 'command_delay_s', round(values['command_delay_s'], 6))
        return values


class Cable(Stage):
    KEYS = ('EI_n_m2', 'Cb_n_m2_s', 'normal_drag_coefficient', 'external_drag_s_inv')
    writes = (PHYSICS,)

    def current(self):
        cable = _json(PHYSICS)['cable']
        return {'bending stiffness EI (N m²)': cable['EI_n_m2'], 'bending damping Cb (N m² s)': cable['Cb_n_m2_s'],
                'normal drag Cd': cable['normal_drag_coefficient'], 'linear drag (1/s)': cable['external_drag_s_inv']}

    def summary(self, r):
        """Least squares on short segments (scripts/fit_cable_segments.py)."""
        return self._segments_summary(r)

    def _segments_summary(self, r):
        """Least squares on short segments (scripts/fit_cable_segments.py)."""
        start, best, u = r['start'], r['best'], r['uncertainty']
        sd = u['standard_deviation']
        ci = [f'×/÷ {10**(1.96*sd[0]):.3g}', f'×/÷ {10**(1.96*sd[1]):.3g}', f'± {1.96*sd[2]:.2g}', f'×/÷ {10**(1.96*sd[3]):.3g}']
        rows = [[name, _mm(start['per_flight_rms_m'].get(name)), _mm(value)] for name, value in best['per_flight_rms_m'].items()]
        headline = [('marker error, 1 s segments (mm)', _mm(start['rms_m']), _mm(best['rms_m']))]
        if best.get('held_out'):
            headline.append(('held out (mm)', None, _mm(best['held_out']['rms_m'])))
        strong = [f'{a} / {b} {c:+.2f}' for a, b, c in _strong(u)]
        parameters = [(n, start['values'][k], best['values'][k], c) for n, k, c in zip(self.current(), self.KEYS, ci)]
        names = r['parameters']
        for i in range(len(self.KEYS), len(names)):          # residual terms, in their search units
            parameters.append((names[i], r['start']['x'][i], best['x'][i], f'± {1.96*sd[i]:.2g}'))
        notes = [f'Loss: {r["loss"]}.', 'Intervals: 95 %, clustered by flight, linearised.' +
                 (f' Strongly correlated: {"; ".join(strong)}.' if strong else ''),
                 f"Failed segment replays during the search: {r['counts']['failed_segment_replays']}; "
                 f"reproduced twice: {best['reproduced_twice']}"]
        if r.get('terms'):
            notes.insert(0, f"Residual terms {', '.join(r['terms'])} ({r['frame']} frame), fitted with the physics.")
        if best.get('held_out') and best['held_out'].get('start_rms_m') is not None:
            headline.append(('held out at the start (mm)', None, _mm(best['held_out']['start_rms_m'])))
        return dict(headline=headline, parameters=parameters, flights=dict(columns=['flight', 'error before', 'error after'], rows=rows),
                    notes=notes)

    def adopt(self, r):
        """The physical values, and the residual exactly as fitted: a result without residual terms
        removes any residual the configuration had, so the adopted model is the one that was scored."""
        physics = _json(PHYSICS)
        values = {k: float(r['best']['values'][k]) for k in self.KEYS}
        physics['cable'].update(values)
        residual = r['adopt'].get('residual')
        if residual is None:
            physics['cable'].pop('residual', None)
        else:
            physics['cable']['residual'] = residual
        _write_json(PHYSICS, physics)
        return dict(values, **({'residual': residual} if residual else {}))


class Coupled(Stage):
    def summary(self, r):
        """scripts/validate_coupled.py: pooled and per-flight errors, fit and held-out flights."""
        results = r['results']
        corrected = r.get('bias_cancelled') or {}
        headline, rows = [], []
        for role, label in (('fit', 'fit flights'), ('held_out', 'held-out flights')):
            s, c = results.get(role), corrected.get(role)
            if not s:
                continue
            # first column: as replayed; second: with the flight's drone bias cancelled
            headline += [(f'{label}: drone position (mm)', _mm(s['position_m']), _mm(c['position_m']) if c else None),
                         (f'{label}: tilt (deg)', s['tilt_deg'], c['tilt_deg'] if c else None),
                         (f'{label}: cable shape (mm)', _mm(s['shape_m']), _mm(c['shape_m']) if c else None),
                         (f'{label}: tip (mm)', _mm(s['tip_m']), _mm(c['tip_m']) if c else None)]
            for name, v in (c or s)['per_flight'].items():
                rows.append([f'{name} ({"held out" if role == "held_out" else "fit"})', _mm(v['position_m']), v['tilt_deg'],
                             _mm(v['shape_m']), _mm(v['tip_m'])])
        stroke = (r.get('stroke_test') or {}).get('summary', {})
        for role, label in (('fit', 'fit strokes'), ('held_out', 'held-out strokes')):
            s = stroke.get(role)
            if s:
                headline += [(f'{label}: tip peak speed, simulated/measured', None, s['peak_ratio_median']),
                             (f'{label}: tip error over the stroke (mm)', None, _mm(s['tip_path_m']))]
        notes = [f'Nothing is fitted: the configured model on {r["horizon_s"]:g} s segments from the sent commands; '
                 'drone position with one bias per flight removed, tilt with one offset per flight removed.']
        if corrected:
            notes.append('Headline columns: as replayed, then with each flight\'s drone bias cancelled in the flight '
                         'controller (the flight table shows the latter). Stroke test: 0.6 s from each stroke onset of the '
                         'decay, strokes and whip flights, bias cancelled.')
        if r.get('cable_residual'):
            notes.append(f"Cable residual in the model ({r['cable_residual'].get('frame', 'body')} frame).")
        return dict(headline=headline, parameters=[],
                    flights=dict(columns=['flight', 'drone position (mm)', 'tilt (deg)', 'cable shape (mm)', 'tip (mm)'], rows=rows),
                    notes=notes)


STAGES = [
    ThrustMap('thrust_map', 'Thrust map', 'Motor PWM to thrust, from the motor logs of the drone-only flights.',
              'scripts/fit_thrust_map.py', [], 'drone-only flights with motor logs', 'log ratio'),
    Drone('drone', 'Drone dynamics', 'Roll/pitch inertia, motor response and air drag by least squares on 1 s segments '
          'of the drone-only flights restarted from the measured state, one constant bias per flight removed; the '
          'command delay profiled on a grid; the measured motor arm is kept.',
          'scripts/fit_drone_segments.py', [], 'drone-only flights', 'm'),
    Cable('cable', 'Cable', 'Bending stiffness and damping and air drag by least squares on 1 s segments, the measured '
          'drone pose driving the clamp and the cable starting from the measured markers.',
          'scripts/fit_cable_segments.py', [], 'cable flights', 'm'),
    Coupled('coupled', 'Coupled validation', 'The whole model on 1 s segments of the cable flights, from the sent '
            'commands, as replayed and with each flight\'s drone bias cancelled, and the stroke test; nothing is '
            'fitted. Reported for fit and held-out flights.',
            'scripts/validate_coupled.py', [], 'cable flights', 'm', adoptable=False),
]
BY_KEY = {s.key: s for s in STAGES}


def stage_of(result_path):
    """The stage a result file belongs to (by its folder under runs/fits)."""
    return BY_KEY.get(Path(result_path).parent.name)
