"""Coupled residuals: the full model (drone driven by the sent commands, carrying the cable) replayed
on 1 s segments of the cable flights from the measured drone state and markers.

Three residuals per frame, each normalised by its RMS at a reference point so they count equally:
drone position (one constant acceleration bias per flight and axis removed, as in the drone fit),
drone tilt (one constant roll/pitch offset per flight removed) and cable shape (each seen marker
relative to the drone's tracking origin; no offset removed). Used by the coupled sensitivity scan
and the coupled refinement stage. Parameters are applied to a configuration by `apply`:

    com_z           drone centre of mass below the tracking origin (m, negative down)
    attachment_z    cable attachment below the tracking origin (m, negative down)
    log10_EI, log10_Cb, Cd, log10_drag   the cable's bending stiffness, bending damping,
                    normal drag coefficient and linear air drag
"""
from copy import deepcopy
import math

import numpy as np

from aerowhip.fitting import flight_drone
from aerowhip.fitting.calibration_set import flight_windows, folders, is_held_out
from aerowhip.fitting.least_squares import FAILED_RESIDUAL_M, ProcessRunner, bias_profile
from aerowhip.takes.calibration_flights import load

KEYS = ('com_z', 'attachment_z', 'log10_EI', 'log10_Cb', 'Cd', 'log10_drag')
WORDS = {'com_z': 'centre of mass height (m)', 'attachment_z': 'attachment height (m)', 'log10_EI': 'log10 EI',
         'log10_Cb': 'log10 Cb', 'Cd': 'normal drag Cd', 'log10_drag': 'log10 linear drag'}


def segments(horizon, setup='cf_2'):
    """({name: [(window, flight)]} fit flights, {...} held out) of the cable flights, with markers."""
    train, held = {}, {}
    for name, folder in folders(setup).items():
        flight = load(folder)
        items = [(w, flight) for w in flight_windows(folder, duration_s=horizon, stride_s=horizon)]
        (held if is_held_out(folder) else train)[name] = items
    return train, held


def current(config):
    """The parameter values a configuration holds."""
    cable = config['cable']
    return dict(com_z=float(config['drone']['center_of_mass_tracking_m'][2]),
                attachment_z=float(config['attachment']['position_tracking_m'][2]),
                log10_EI=math.log10(cable['EI_n_m2']), log10_Cb=math.log10(cable['Cb_n_m2_s']),
                Cd=float(cable['normal_drag_coefficient']), log10_drag=math.log10(cable['external_drag_s_inv']))


def apply(config, **values):
    """A copy of the configuration with these parameters set (keys of KEYS)."""
    c = deepcopy(config)
    for key, v in values.items():
        if key == 'com_z':
            c['drone']['center_of_mass_tracking_m'][2] = float(v)
        elif key == 'attachment_z':
            c['attachment']['position_tracking_m'][2] = float(v)
        elif key == 'log10_EI':
            c['cable']['EI_n_m2'] = 10.**v
        elif key == 'log10_Cb':
            c['cable']['Cb_n_m2_s'] = 10.**v
        elif key == 'Cd':
            c['cable']['normal_drag_coefficient'] = float(v)
        elif key == 'log10_drag':
            c['cable']['external_drag_s_inv'] = 10.**v
        else:
            raise KeyError(key)
    return c


def _replay(parameters):
    config, firmware = parameters
    replay = flight_drone.DroneReplay(config, firmware)
    return lambda item: replay.coupled_errors(*item[1], pipeline=True)


class Residuals:
    """groups: {flight name: [(window, flight)]}."""

    def __init__(self, config, firmware, groups):
        self.config, self.firmware, self.groups = config, firmware, groups
        self.names = list(groups)
        self.items = [(name, i) for name in self.names for i in self.groups[name]]
        self.runner = ProcessRunner(self.items, _replay)
        self.scale = (1., 1., 1.)            # position (m), tilt (rad), shape (m)

    def set_scales(self, **values):
        """Scales that make each component's residuals sum to the frame count at this point, so the
        loss is 1 there: the shape scale includes sqrt(seen markers per frame), since a frame has
        ten shape entries (zero where unseen) but three position and two tilt entries."""
        self.scale = (1., 1., 1.)
        _, _, report = self(**values)
        self.scale = (report['position_m'], math.radians(report['tilt_deg']),
                      report['shape_m']*math.sqrt(report['seen_per_frame']))

    def errors(self, **values):
        """{name: [(N, 15) or None]}, {name: [messages]}."""
        results = self.runner((apply(self.config, **values), self.firmware))
        errors, messages = {n: [] for n in self.names}, {n: [] for n in self.names}
        for (name, _), (e, message) in zip(self.items, results):
            errors[name].append(e)
            if message:
                messages[name].append(message)
        return errors, messages

    def __call__(self, **values):
        """(r, failures, report): report has position RMS (m), tilt RMS (deg), shape RMS (m), overall
        and per flight."""
        errors, _ = self.errors(**values)
        sp, st, ss = self.scale
        pieces, failures = [], 0
        report = {k: {} for k in ('per_flight_position_m', 'per_flight_tilt_deg', 'per_flight_shape_m')}
        squares, frames, seen = np.zeros(3), 0, 0
        for name in self.names:
            good = [e for e in errors[name] if e is not None]
            profiled, _ = bias_profile([e[:, :3] for e in good])
            offset = np.concatenate([e[:, 3:5] for e in good]).mean(0) if good else np.zeros(2)
            tilts = [e[:, 3:5]-offset for e in good]
            shapes = [e[:, 5:] for e in good]
            fseen = sum(int(np.isfinite(s).sum()) for s in shapes)
            report['per_flight_position_m'][name] = _rms(profiled)
            report['per_flight_tilt_deg'][name] = math.degrees(_rms(tilts)) if tilts else math.nan
            report['per_flight_shape_m'][name] = math.sqrt(sum(float(np.nansum(s**2)) for s in shapes)/fseen) if fseen else math.nan
            it = iter(zip(profiled, tilts, shapes))
            for e, (w, _) in zip(errors[name], self.groups[name]):
                n = len(w.time)-1
                if e is None:
                    failures += 1
                    pieces.append(np.full(15*n, FAILED_RESIDUAL_M/min(self.scale)))
                else:
                    pos, tilt, shape = next(it)
                    mask = np.isfinite(shape)
                    squares += [np.sum(pos**2), np.sum(tilt**2), float(np.nansum(shape**2))]
                    frames += n
                    seen += int(mask.sum())
                    pieces.append(np.r_[pos.ravel()/sp, tilt.ravel()/st, np.where(mask, shape, 0.).ravel()/ss])
        report['position_m'] = float(np.sqrt(squares[0]/frames)) if frames else math.nan
        report['tilt_deg'] = math.degrees(np.sqrt(squares[1]/frames)) if frames else math.nan
        report['shape_m'] = float(np.sqrt(squares[2]/seen)) if seen else math.nan
        report['seen_per_frame'] = seen/frames if frames else math.nan
        return np.concatenate(pieces), failures, report

    def value(self, r):
        """The loss: sqrt of the mean over the three components of (RMS/scale)², 1 at the reference
        point (each component's residuals sum to the frame count there)."""
        return float(np.sqrt(np.sum(r**2)/(3*len(r)/15)))

    def flight_groups(self):
        groups, start = [], 0
        for name in self.names:
            size = sum(15*(len(w.time)-1) for w, _ in self.groups[name])
            groups.append(np.arange(start, start+size))
            start += size
        return groups


def _rms(arrays):
    if not arrays:
        return math.nan
    e = np.concatenate(arrays)
    return float(np.sqrt(np.mean(np.sum(e**2, axis=1))))
