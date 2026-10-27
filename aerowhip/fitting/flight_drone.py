"""Drone replay on calibration flights: the sent acceleration commands drive the native
flight pipeline (FlightController: firmware thrust vector, thrust map, attitude loop) open
loop from the measured state at the window start.

Two losses on the tracking-position error d(t) over a window: the raw RMS, and the RMS after
removing the best constant acceleration bias per window and axis (d - b t^2/2). Over 2 s the
raw error is dominated by slowly varying biases of about 0.1 m/s^2 (attitude-estimate offsets,
airflow) that the real host PID absorbs; the debiased loss measures the modelled dynamics.

Drone-only flights use the model with a negligible cable (one 2 cm link, 1 mg); command times are the
frame index of the pose each command was computed from (the windows' uniform 100 Hz clock).
"""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import math
import os
import tomllib

import numpy as np

from aerowhip.fitting.flight_cable import endpoint_velocity
from aerowhip.paths import ROOT
from aerowhip.physics.model import Model, native_module

FIRMWARE = ROOT/'configs/controllers/crazyflie_firmware.toml'    # the flown firmware (law 'firmware'); crazyflie_flight.toml is the identified fallback


def firmware_config(path=FIRMWARE):
    return tomllib.loads(path.read_text())


def drone_only(config):
    """The model without cable: the engine needs one, so a single 2 cm link of 1 mg remains. Its
    effect on the drone is below 0.12 mm over 2 s (a 0.1 g ten-link cable moved the drone by up to
    12 mm), and one link halves the cost of a replay."""
    config = deepcopy(config)
    cable = config['cable']
    cable['total_mass_kg'] = 1e-6
    cable['marker_interval_lengths_m'] = [.02]
    cable['moving_marker_masses_kg'] = [1e-8]
    cable['normal_drag_coefficient'] = 0.
    config['simulation'] = dict(config['simulation'], segments_per_marker_interval=[1])
    return config


def body_rate(time, rotations):
    """Body angular velocity at the newest sample from the causal rotation history."""
    R = rotations
    logs = []
    for Ri in R:
        D = R[-1].T@Ri
        angle = math.acos(np.clip((np.trace(D)-1)/2, -1, 1))
        v = np.array([D[2, 1]-D[1, 2], D[0, 2]-D[2, 0], D[1, 0]-D[0, 1]])/2
        logs.append(v if angle < 1e-9 else v*angle/math.sin(angle))
    return endpoint_velocity(time, np.array(logs))


class DroneReplay:
    def __init__(self, config, firmware, *, delay_s=None):
        self.config, self.firmware = config, firmware
        self.model = Model(config, load_weights=False)
        self.native = native_module()
        self.delay = config['controller']['command_delay_s'] if delay_s is None else delay_s
        self.period = config['controller']['controller_period_s']

    def initial_cable(self, window, tracking_velocity, omega):
        """Full-mesh cable positions/velocities from the measured markers at the window start,
        projected onto the lengths; the root velocity is the rigid-body one of the drone state."""
        R, p = window.history_rotation[-1], window.history_position[-1]
        attachment = np.asarray(self.model.attachment_tracking)
        root = p+R@attachment
        root_v = tracking_velocity+R@np.cross(omega, attachment)
        markers = window.history_markers
        velocities = endpoint_velocity(window.history_time, markers)
        rest = np.asarray(self.model.rest)
        q, qv = [root], [root_v]
        for i in range(len(rest)):
            edge = markers[-1, i]-q[-1]
            t = edge/np.linalg.norm(edge)
            q.append(q[-1]+t*rest[i])
            relative = velocities[i]-qv[-1]
            qv.append(qv[-1]+relative-t*(t@relative))
        return np.array(q), np.array(qv)

    def run(self, window, flight, *, markers=False):
        hp, hR, ht = window.history_position, window.history_rotation, window.history_time
        v, w = endpoint_velocity(ht, hp), body_rate(ht, hR)
        q = qv = None
        if markers:
            q, qv = self.initial_cable(window, v, w)
        state = self.model.native.initial_state(hp[-1], hR[-1], v, w, q, qv)
        # Commands in force: log rows mapped to the frame they were computed from.
        ok = np.isfinite(flight.log_stamp)
        stamps, sent = flight.log_stamp[ok], flight.sent[ok].copy()
        # The sideways trims cancel biases of the real drone (hover offsets) that the model
        # does not have; the vertical trim stays (the thrust map models the thrust error).
        sent[:, :2] -= flight.trim[ok][:, :2]
        frame = np.searchsorted(flight.stamp, stamps-1e-6)
        last = np.r_[frame[1:] != frame[:-1], True]       # one command per frame: the newest
        stamps, sent, frame = stamps[last], sent[last], frame[last]
        first = int(np.searchsorted(flight.stamp, window.start_stamp-1e-6))
        command_time = (frame-first)*.01
        battery = None if flight.battery is None else flight.battery[ok][last]
        inside = (command_time > -.5) & (command_time < window.time[-1])
        voltage = float(np.nanmedian(battery[inside])) if battery is not None and np.isfinite(battery[inside]).any() else self.firmware['reference_voltage_v']
        controller = self.native.FlightController(self.firmware, self.config, voltage)
        yaw = math.atan2(hR[-1][1, 0], hR[-1][0, 0])
        def command(a):
            c = np.zeros(13); c[6:9] = a; c[9] = yaw; return c
        before = np.where(command_time <= 0)[0]
        hold = command(sent[before[-1]] if len(before) else sent[0])
        actuator = deepcopy(self.config['actuator'])
        thrust = controller(state, hold)[0]
        actuator['initial_thrust_n'] = [float(np.clip(thrust/4, actuator['minimum_thrust_n'][i], actuator['maximum_thrust_n'][i])) for i in range(4)]
        runtime = self.native.Runtime(self.model.native, state, hold, self.period, self.delay, self.native.Actuator(actuator))
        events = [(float(t), command(a)) for t, a in zip(command_time, sent) if 0 < t < window.time[-1]]
        positions, cable = [], []
        n = len(self.model.rest)+1
        index = np.asarray(self.model.marker_indices)[1:]
        j = 0
        for k in range(len(window.time)-1):
            start, end = k*.01, (k+1)*.01
            batch = []
            while j < len(events) and events[j][0] < end-1e-9:
                if events[j][0] >= start-1e-9:
                    batch.append((max(events[j][0], runtime.time), events[j][1]))
                j += 1
            runtime.step(batch, controller)
            observation = self.model.native.observation(runtime.state())
            positions.append(observation[:3])
            if markers:
                cable.append(observation[18:18+3*n].reshape(n, 3)[index])
        return (np.array(positions), np.array(cable)) if markers else np.array(positions)

    def errors(self, window, flight):
        """(N, 3) simulated minus measured tracking position."""
        return self.run(window, flight)-window.position[1:]


def debias(error, dt=.01):
    t = (np.arange(len(error))+1)*dt
    basis = .5*t**2
    return error-np.outer(basis, error.T@basis/(basis@basis))


def coupled_errors(replay, window, flight):
    """Drone position error (N, 3) and cable-shape error (N, 10): markers relative to the
    drone's tracking origin, NaN where a marker was not seen."""
    p, markers = replay.run(window, flight, markers=True)
    drone = p-window.position[1:]
    shape = np.linalg.norm((markers-p[:, None])-(window.markers[1:]-window.position[1:, None]), axis=2)
    shape[~window.seen[1:]] = np.nan
    return drone, shape


def evaluate(config, firmware, items, *, threads=None, delay_s=None):
    """items: [(window, flight)]; returns per-window error arrays (None on failure)."""
    replay = DroneReplay(config, firmware, delay_s=delay_s)

    def one(item):
        try:
            return replay.errors(*item)
        except Exception:
            return None
    with ThreadPoolExecutor(threads or os.cpu_count()) as pool:
        return list(pool.map(one, items))


def rms(errors, *, debiased=False):
    """Pooled RMS position error (m); NaN if any window failed."""
    if any(e is None for e in errors):
        return math.nan
    e = np.concatenate([debias(x) if debiased else x for x in errors])
    return float(np.sqrt(np.mean(np.sum(e**2, axis=1))))
