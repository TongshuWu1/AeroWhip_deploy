"""Online identification: a moving-window estimator of the parameters a flight can teach, with a
robust filter so that one-off events (a cable flip, a marker swap, a gust) do not move the estimate.

Parameters (two families, each scored on its own measurement, see docs/data_and_fitting.md):
  cable: log10 EI, log10 Cb, Cd (the cable fit's search coordinates; the linear drag, coupled to Cd,
         and the static residual terms, coupled to each other, stay at the offline fit)
  drone: motor_response_s, body_drag_xy_s_inv, command_delay_s
The prior is the offline fits: their values, standard deviations and correlations.

Every update takes the newest window (`window_s`, 2 s) of the flight:
  cable: CableReplay driven by the measured drone pose, from the measured markers; residual the
         signed marker errors (seen markers only);
  drone: DroneReplay of the sent commands from the measured state and cable (coupled model);
         residual the position error with one constant acceleration removed per window and axis, and
         the roll/pitch error with its window mean removed (attitude-estimate offsets).
For each family: one replay at the estimate and one per parameter (forward differences), in
parallel threads; then one Gauss-Newton step in information form, with the family's covariance as
the prior (a Kalman update), with old evidence forgotten: its weight halves every `half_life_s` (5 min) of
flight, the uncertainty never exceeding the prior's.
The measurement noise is the median residual of the recent accepted windows, starting from the
offline fit's typical residual and never below half of it (a quiet hover window has a small
residual but is not more precise: without the floor it pulled the stiffness away); the
residual is sampled every 50 ms and one window counts as `effective_samples` independent samples
(they are strongly correlated: counting them all made the estimate overconfident within seconds, and
the held-out windows then pooled into false changes). A step is applied only if it lowers the
window's own error (the full step or half of it, both replayed); where the linearisation fails, the
window is not used.

Robust filter (the user's rule: a flip or a sudden change counts only if it keeps happening):
  gated   the window is not used: outside the recorded motion (takeoff, landing, settling, safety
          landings), the cable touching the drone (aerowhip.takes.contact), drone tilt over 60 deg, too
          few markers (no window), a failed replay;
  outlier the window's step is improbable under the current uncertainty (Mahalanobis distance
          above the 99.9 % chi-square bound) or the model cannot explain it (its residual is more
          than `unexplained` times the recent median): it is held, not applied;
  change  when `persist_k` of the last `persist_m` windows of a family are outliers whose steps agree
          in direction, they are pooled and applied as a change point (the covariance widened by
          the pooled step first); outliers that do not recur expire.
"""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass
import math
import time

import numpy as np

from aerowhip.fitting import flight_drone
from aerowhip.fitting.flight_cable import CableReplay
from aerowhip.physics.model import Model
from aerowhip.physics.model_configuration import fit_configuration
from aerowhip.planning.ensemble import Ensemble
from aerowhip.takes.calibration_flights import windows

SKIP_PHASES = ('takeoff', 'land', 'settle', 'link-', 'hover-cycle-land')
CHI2_999 = {1: 10.83, 2: 13.82, 3: 16.27, 4: 18.47, 5: 20.52, 6: 22.46}
SUBSAMPLE = 5                     # frames (10 ms) between residual samples
BOUNDS = {'motor_response_s': (1e-3, .08), 'body_drag_xy_s_inv': (0., 3.)}   # physically possible values
HALF_LIFE_S = 300.                # flight time over which old evidence loses half its weight (user, 2026-10-28)
POSITION_SCALE_M, TILT_SCALE_RAD = .01, .01


@dataclass
class Parameter:
    name: str
    family: str
    value: float
    sd: float
    step: float
    low: float
    high: float


class ModelSpace:
    """The online parameters and the model configuration at any of their values."""
    CABLE = ('log10 EI', 'log10 Cb', 'Cd')
    DRONE = ('motor_response_s', 'body_drag_xy_s_inv')

    def __init__(self, base_config, cable_fit, drone_fit, *, delay_sd=.005):
        self.base = fit_configuration(deepcopy(base_config))
        self.ensemble = Ensemble(self.base, cable_fit, drone_fit)
        self.cable_x = np.asarray(cable_fit['best']['x'], float)
        self.drone_x = np.asarray(drone_fit['best']['x'], float)
        cnames, csd = cable_fit['parameters'], np.asarray(cable_fit['uncertainty']['standard_deviation'], float)
        dnames, dsd = drone_fit['parameters'], np.asarray(drone_fit['uncertainty']['standard_deviation'], float)
        self.cable_index = [cnames.index(n) for n in self.CABLE]
        self.drone_index = [dnames.index(n) for n in self.DRONE]
        self.parameters = []
        for n, i in zip(self.CABLE, self.cable_index):
            self.parameters.append(Parameter(n, 'cable', float(self.cable_x[i]), float(csd[i]), float(csd[i])/2, -np.inf, np.inf))
        self.parameters[-1].low = .05                                 # Cd
        for n, i in zip(self.DRONE, self.drone_index):
            low, high = BOUNDS[n]
            self.parameters.append(Parameter(n, 'drone', float(self.drone_x[i]), float(dsd[i]), float(dsd[i])/2, low, high))
        delay = float(drone_fit['best'].get('command_delay_s', self.base['controller']['command_delay_s']))
        self.parameters.append(Parameter('command_delay_s', 'drone', delay, delay_sd, .004, 0., .04))
        # prior covariance: the fits' correlations within each family (the delay independent)
        # typical residual of the offline fits, in each family's residual units: the measurement noise
        # until the flight's own accepted windows give it (and half of it as its floor)
        self.typical = dict(cable=float(cable_fit['best']['rms_m'])/math.sqrt(3),       # per signed component
                            drone=math.sqrt(((drone_fit['best']['rms_m']/POSITION_SCALE_M)**2
                                             +(math.radians(drone_fit['best']['tilt_deg'])/TILT_SCALE_RAD)**2)/5))
        self.prior = np.diag([p.sd**2 for p in self.parameters])
        for fit, index, offset in ((cable_fit, self.cable_index, 0), (drone_fit, self.drone_index, len(self.CABLE))):
            corr = np.asarray(fit['uncertainty'].get('correlation', np.eye(len(fit['parameters']))), float)
            for a, i in enumerate(index):
                for b, j in enumerate(index):
                    self.prior[offset+a, offset+b] = corr[i, j]*self.parameters[offset+a].sd*self.parameters[offset+b].sd

    @property
    def names(self):
        return [p.name for p in self.parameters]

    def family(self, name):
        return [k for k, p in enumerate(self.parameters) if p.family == name]

    def config(self, theta, *, cable=True):
        theta = np.asarray(theta, float)
        x = self.cable_x.copy()
        x[self.cable_index] = theta[:len(self.CABLE)]
        c = self.ensemble.cable_config(x)
        d = self.drone_x.copy()
        d[self.drone_index] = theta[len(self.CABLE):len(self.CABLE)+len(self.DRONE)]
        delay = float(theta[-1])
        c = flight_drone.vehicle(c, [d[0], 1., d[1], delay], cable=cable)
        c['drone']['mass_kg'] = float(self.base['drone']['mass_kg'])   # the base model's drone, not the manifest's
        c['drone']['body_drag_s_inv'] = [float(d[2]), float(d[2]), float(d[3])]
        return c


def firmware_for(params):
    """The firmware law at the gains the flight recorded (cf_6 flights record them; older flights
    used the firmware file's values, as the offline fits did)."""
    firmware = flight_drone.firmware_config()
    for key, value in (params.get('attitude') or {}).items():
        group, name = key.split('.', 1)
        if group in firmware and isinstance(firmware[group], dict):
            firmware[group][name] = float(value)
    return firmware


def cable_residual(config, window):
    """Signed marker errors (x, y, z), NaN where a marker was not seen. Signed, not distances: a
    distance's slope flips where the simulated marker passes the measured one, and the linearised
    step then points the wrong way."""
    replay = CableReplay(Model(config).native)
    e = replay.run(window)-window.markers[1:]
    e[~window.seen[1:]] = np.nan
    return e[SUBSAMPLE-1::SUBSAMPLE].ravel()


def drone_residual(config, firmware, window, flight, coupled=True):
    """coupled: the model with the cable, started from the measured markers; otherwise drone only."""
    replay = flight_drone.DroneReplay(config, firmware)
    if coupled:
        p, R, _, _ = replay.run(window, flight, markers=True, attitude=True, pipeline=True)
    else:
        p, R = replay.run(window, flight, attitude=True, pipeline=True)
    position = flight_drone.debias(p-window.position[1:])
    tilt = flight_drone.tilt_error(R, window.rotation[1:])
    tilt = tilt-tilt.mean(0)
    s = slice(SUBSAMPLE-1, None, SUBSAMPLE)
    return np.r_[(position[s]/POSITION_SCALE_M).ravel(), (tilt[s]/TILT_SCALE_RAD).ravel()]


def window_error(name, now, offline):
    """The error of a window at the estimate it arrived to (a prediction: the window has not updated it
    yet) and at the offline model, on the same samples. cable: RMS 3-D marker error over every seen
    marker, mm, overall and per marker 1..10; drone: the normalised residual's RMS."""
    now, offline = np.asarray(now, float), np.asarray(offline, float)
    if name == 'cable':
        a, b = now.reshape(-1, 10, 3), offline.reshape(-1, 10, 3)
        both = np.isfinite(a).all(2) & np.isfinite(b).all(2)
        def rms(e, sel):
            return float(np.sqrt(np.mean(np.sum(e[sel]**2, axis=-1))))*1000 if sel.any() else None
        return dict(unit='mm', now=rms(a, both), offline=rms(b, both),
                    markers_now=[rms(a[:, k], both[:, k]) for k in range(10)],
                    markers_offline=[rms(b[:, k], both[:, k]) for k in range(10)])
    both = np.isfinite(now) & np.isfinite(offline)
    return dict(unit='', now=float(np.sqrt(np.mean(now[both]**2))), offline=float(np.sqrt(np.mean(offline[both]**2))))


class FamilyState:
    def __init__(self, index):
        self.index = index
        self.history = []            # statuses of the recent windows ('accepted', 'outlier', 'gated')
        self.pending = []            # held outliers: dict(window, theta, info, gradient, step)
        self.rms = []                # residual RMS at the estimate of recently accepted windows
        self.counts = dict(accepted=0, outlier=0, change=0, no_gain=0, gated=0)


class OnlineEstimator:
    def __init__(self, space, *, window_s=2., effective_samples=8., half_life_s=HALF_LIFE_S, unexplained=3., persist_k=4, persist_m=6,
                 threads=8, families=('cable', 'drone')):
        self.space = space
        self.theta = np.array([p.value for p in space.parameters], float)
        self.P = space.prior.copy()
        self.window_s, self.effective, self.half_life = window_s, effective_samples, half_life_s
        self.unexplained, self.persist_k, self.persist_m = unexplained, persist_k, persist_m
        self.pool = ThreadPoolExecutor(threads)
        self.families = {f: FamilyState(space.family(f)) for f in families}
        self.last_stamp = None
        self.updates = 0

    # ------------------------------------------------------------------ the window
    def latest_window(self, flight):
        """The newest window ending at the flight's last frames, or (None, reason)."""
        ws = windows(flight, duration_s=self.window_s, stride_s=.05, min_seen=8)
        if not ws:
            return None, 'no usable window (hidden markers or dropped frames)'
        w = ws[-1]
        end = w.start_stamp+w.time[-1]
        if flight.stamp[-1]-end > .3:
            return None, 'no usable window at the newest frames'
        return w, None

    def gate(self, window, flight, has_cable):
        phases = {flight.phase_at(window.start_stamp), flight.phase_at(window.start_stamp+window.time[-1])}
        for ph in phases:
            if ph.startswith(SKIP_PHASES):
                return f'phase {ph}'
        tilt = np.degrees(np.arccos(np.clip(window.rotation[:, 2, 2], -1, 1))).max()
        if tilt > 60:
            return f'tilt {tilt:.0f} deg'
        if has_cable:                      # the user's rule: the cable touching the drone (not rising above the clamp)
            from aerowhip.takes.contact import clearance, contact_events
            attachment = np.asarray(self.space.base['attachment']['position_tracking_m'], float)
            arc = np.cumsum(self.space.base['cable']['marker_interval_lengths_m'])
            clear = clearance(window.markers, window.position, window.rotation, attachment, arc)
            if contact_events(clear):
                return f'cable touching the drone ({np.nanmin(clear)*100:+.0f} cm)'
        return None

    # ------------------------------------------------------------------ one update
    def step(self, flight, params, has_cable=True):
        """One update from the newest window of `flight` (a CalibrationFlight of the recent part;
        `params` its parameters.json). Returns the record of what happened (also for gated windows)."""
        t0 = time.perf_counter()
        record = dict(stamp=float(flight.stamp[-1]), phase=flight.phases[-1] if flight.phases else '', families={})
        window, reason = self.latest_window(flight)
        if window is not None:
            reason = self.gate(window, flight, has_cable)
            record['phase'] = window.phase
        if self.last_stamp is not None:
            dt = max(0., record['stamp']-self.last_stamp)
            self.forget(dt)
        self.last_stamp = record['stamp']
        if reason is not None:
            for name, fam in self.families.items():
                self._push(fam, 'gated')
                record['families'][name] = dict(status='gated', reason=reason)
        else:
            firmware = firmware_for(params)
            jobs = {}
            for name, fam in self.families.items():
                if name == 'cable' and not has_cable:
                    continue
                if name == 'cable':
                    def submit(th):
                        return self.pool.submit(cable_residual, self.space.config(th, cable=has_cable), window)
                else:
                    def submit(th):
                        return self.pool.submit(drone_residual, self.space.config(th, cable=has_cable), firmware, window, flight, has_cable)
                thetas = [self.theta]+[self.theta+np.eye(len(self.theta))[k]*self.space.parameters[k].step for k in fam.index]
                offline = np.array([q.value for q in self.space.parameters])
                jobs[name] = (submit, [submit(th) for th in thetas], submit(offline))
            for name, (submit, futures, reference) in jobs.items():
                record['families'][name] = self._update(name, futures, submit)
                try:                       # the window's error before it updates anything, and the offline model's
                    record['families'][name]['error'] = window_error(name, futures[0].result(), reference.result())
                except Exception:
                    pass
            for name in self.families:
                if name not in record['families']:
                    record['families'][name] = dict(status='skipped', reason='no cable on this flight')
        self.updates += 1
        record.update(theta={n: float(v) for n, v in zip(self.space.names, self.theta)},
                      sd={n: float(math.sqrt(max(self.P[k, k], 0.))) for k, n in enumerate(self.space.names)},
                      seconds=round(time.perf_counter()-t0, 3))
        return record

    def forget(self, dt):
        """Old evidence loses weight: the information halves every `half_life` s of flight (the
        covariance doubles), never beyond the prior's uncertainty."""
        if dt <= 0 or not np.isfinite(self.half_life):
            return
        P = self.P*2.**(dt/self.half_life)
        scale = np.sqrt(np.minimum(1., np.diag(self.space.prior)/np.maximum(np.diag(P), 1e-300)))
        self.P = P*np.outer(scale, scale)

    def _push(self, fam, status):
        fam.history = (fam.history+[status])[-self.persist_m:]
        fam.counts[status if status in fam.counts else 'gated'] += 1
        fam.pending = [p for p in fam.pending if p['age'] < self.persist_m]
        for p in fam.pending:
            p['age'] += 1

    def _update(self, name, futures, submit):
        fam = self.families[name]
        try:
            results = [f.result() for f in futures]
        except Exception as exc:                       # a replay failed: the window says nothing
            self._push(fam, 'gated')
            return dict(status='gated', reason=f'replay failed: {type(exc).__name__}: {exc}'[:160])
        r0 = results[0]
        mask = np.isfinite(r0)
        for r in results[1:]:
            mask &= np.isfinite(r)
        if mask.sum() < 20:
            self._push(fam, 'gated')
            return dict(status='gated', reason='too few residual samples')
        r0 = r0[mask]
        idx = fam.index
        steps = np.array([self.space.parameters[k].step for k in idx])
        J = np.column_stack([(r[mask]-r0)/h for r, h in zip(results[1:], steps)])
        rms0 = float(np.sqrt(np.mean(r0**2)))
        typical = self.space.typical[name]
        s2 = float(np.median(np.square(fam.rms[-20:]))) if len(fam.rms) >= 3 else typical**2
        s2 = max(s2, (.5*typical)**2)            # a quiet window (hover) is not a precise one
        weight = self.effective/len(r0)          # the samples of one window are strongly correlated
        info = weight*(J.T@J)/s2
        gradient = weight*(J.T@(-r0))/s2
        P = self.P[np.ix_(idx, idx)]
        P_inv = np.linalg.inv(P)
        post = np.linalg.inv(P_inv+info)
        step = post@gradient
        m2 = float(step@P_inv@step)
        bound = CHI2_999[len(idx)]
        explained = len(fam.rms) < 3 or rms0 <= self.unexplained*math.sqrt(s2)
        out = dict(rms=rms0, noise=math.sqrt(s2), mahalanobis2=m2, samples=int(mask.sum()))
        predicted = float(r0@r0-np.sum((r0+J@step)**2))/max(float(r0@r0), 1e-300)
        if m2 <= bound and explained and predicted < .01:
            # already at this window's best fit: it confirms the estimate (its information counts)
            self._apply(idx, step, post)
            fam.rms = (fam.rms+[rms0])[-50:]
            self._push(fam, 'accepted')
            out.update(status='accepted', step=step.tolist(), converged=True)
            return out
        if m2 <= bound and explained:
            # the step must lower this window's own error (the full step or half of it), else the
            # linearisation does not hold here and the window is not used
            trials = []
            for f in (1., .5):
                th = self.theta.copy(); th[idx] += f*step
                trials.append((f, submit(th)))
            best = None
            for f, fut in trials:
                try:
                    r = fut.result()[mask]
                except Exception:
                    continue
                if np.isfinite(r).all():
                    rms = float(np.sqrt(np.mean(r**2)))
                    if rms < rms0 and (best is None or rms < best[1]):
                        best = (f, rms)
            if best is None:
                self._push(fam, 'no_gain')
                out.update(status='no gain', reason=f'the step promised {predicted*100:.0f} % less error and gave none')
                return out
            f, rms1 = best
            self._apply(idx, f*step, post)
            fam.rms = (fam.rms+[rms0])[-50:]
            self._push(fam, 'accepted')
            out.update(status='accepted', step=(f*step).tolist(), rms_after=rms1, fraction=f)
            return out
        why = 'improbable step' if m2 > bound else f'not explained by the model (residual {rms0/math.sqrt(s2):.1f}x the recent median)'
        fam.pending.append(dict(age=0, theta=self.theta[idx].copy(), info=info, gradient=gradient, step=step, why=why))
        self._push(fam, 'outlier')
        out.update(status='outlier', reason=why, step=step.tolist())
        change = self._persistent(fam, idx, P, submit, mask, rms0)
        if change:
            out.update(status='change', reason=f'{change} outliers agree: applied as a change', step=None)
            fam.counts['change'] += 1
        elif change == 0:
            out.update(reason=why+'; the agreeing outliers did not lower the error: dropped')
        return out

    def _persistent(self, fam, idx, P, submit, mask, rms0):
        """Apply the held outliers as a change point when they keep happening and agree."""
        if fam.history.count('outlier') < self.persist_k or len(fam.pending) < self.persist_k:
            return None
        L = np.linalg.cholesky(np.linalg.inv(P))           # whitening: directions in the prior's metric
        dirs = np.array([L.T@p['step'] for p in fam.pending])
        dirs /= np.maximum(np.linalg.norm(dirs, axis=1, keepdims=True), 1e-12)
        mean = dirs.mean(0)
        if np.linalg.norm(mean) < 1e-9:
            return None
        agree = int((dirs@(mean/np.linalg.norm(mean)) > .5).sum())
        if agree < self.persist_k:
            return None
        # pooled information of the held windows, each linearised at its own estimate
        info = sum(p['info'] for p in fam.pending)
        b = sum(p['gradient']+p['info']@p['theta'] for p in fam.pending)
        pooled_step = np.linalg.solve(info+np.linalg.inv(P), b+np.linalg.inv(P)@self.theta[idx])-self.theta[idx]
        wide = P+np.diag(pooled_step**2)                  # the change point: the old certainty no longer holds
        wide_inv = np.linalg.inv(wide)
        post = np.linalg.inv(wide_inv+info)
        target = post@(b+wide_inv@self.theta[idx])
        # like every step, the change must lower the newest window's error (pooled outliers once drove
        # the motor response to 0.3 s on a drone-only strokes flight)
        theta = self.theta.copy(); theta[idx] = target
        for k in idx:
            q = self.space.parameters[k]
            theta[k] = float(np.clip(theta[k], q.low, q.high))
        try:
            r = submit(theta).result()[mask]
            better = bool(np.isfinite(r).all() and np.sqrt(np.mean(r**2)) < rms0)
        except Exception:
            better = False
        fam.pending.clear()
        fam.history = []
        if not better:
            return 0
        self._apply(idx, theta[idx]-self.theta[idx], post)
        return agree

    def _apply(self, idx, step, post):
        theta = self.theta.copy()
        theta[idx] += step
        for k in idx:
            p = self.space.parameters[k]
            theta[k] = float(np.clip(theta[k], p.low, p.high))
        self.theta = theta
        self.P[np.ix_(idx, idx)] = post
        for k in idx:                                      # no cross terms between the families' blocks
            others = [j for j in range(len(self.theta)) if j not in idx]
            self.P[k, others] = 0.
            self.P[others, k] = 0.

    def summary(self):
        return dict(theta={n: float(v) for n, v in zip(self.space.names, self.theta)},
                    sd={n: float(math.sqrt(self.P[k, k])) for k, n in enumerate(self.space.names)},
                    prior={p.name: dict(value=p.value, sd=p.sd) for p in self.space.parameters},
                    counts={name: fam.counts for name, fam in self.families.items()}, updates=self.updates)

    def close(self):
        self.pool.shutdown(wait=False, cancel_futures=True)

