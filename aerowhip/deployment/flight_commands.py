"""ROS command arbitration and high-level polynomials, separate from physics."""
import math
import numpy as np

from aerowhip.deployment.custom_simulation import StreamingCommands
from aerowhip.deployment.ros_bridge import vector


def validate_plan(plan):
    pieces = plan.get('pieces')
    if not isinstance(pieces, list) or not 1 <= len(pieces) <= 64:
        raise ValueError('Trajectory must contain 1..64 polynomial pieces')
    clean = []
    for piece in pieces:
        duration = piece.get('duration_s')
        if type(duration) not in (int, float) or not math.isfinite(duration) or duration <= 0:
            raise ValueError('Polynomial durations must be positive and finite')
        coefficients = piece.get('coefficients')
        if not isinstance(coefficients, list) or len(coefficients) != 4:
            raise ValueError('A piece needs XYZ/yaw polynomials')
        clean.append(dict(duration_s=float(duration), coefficients=[vector(c, 8) for c in coefficients]))
    scale = plan.get('timescale', 1.)
    if type(scale) not in (int, float) or not math.isfinite(scale) or scale <= 0:
        raise ValueError('Trajectory timescale must be positive and finite')
    if type(plan.get('reverse', False)) is not bool:
        raise ValueError('Trajectory reverse must be boolean')
    return dict(pieces=clean, timescale=float(scale), reverse=plan.get('reverse', False),
                shift=vector(plan.get('shift', [0., 0., 0.]), 3))


def polynomial_reference(plan, elapsed_s):
    """Firmware XYZ/yaw polynomial contract, evaluated at native output times.

    Rate feedforward follows firmware pptraj.c flatness with GRAV=9.81; it is
    distinct from measured body rates and does not select a firmware controller.
    """
    scale = plan['timescale']
    total = sum(p['duration_s'] for p in plan['pieces'])
    elapsed = max(0., elapsed_s / scale)
    finished = elapsed >= total
    local = min(elapsed, total)
    if plan['reverse']:
        local = total-local
    piece = plan['pieces'][-1]
    for candidate in plan['pieces']:
        piece = candidate
        if local <= piece['duration_s']:
            break
        local -= piece['duration_s']
    local = min(local, piece['duration_s'])
    derivatives = []
    coefficients = np.asarray(piece['coefficients'], dtype=np.float64)
    sign = -1. if plan['reverse'] else 1.
    for order in range(4):
        derivatives.append(np.polynomial.polynomial.polyval(local, coefficients.T) * (sign/scale)**order)
        coefficients = coefficients[:, 1:] * np.arange(1, coefficients.shape[1])
    p, v, a, jerk = derivatives
    command = np.zeros(13)
    command[:3], command[9] = p[:3]+plan['shift'], p[3]
    if finished:
        return command
    command[3:6], command[6:9] = v[:3], a[:3]
    thrust = a[:3]+[0., 0., 9.81]
    norm = np.linalg.norm(thrust)
    if norm <= 1e-9:
        raise ValueError('High-level trajectory has singular flatness thrust')
    z = thrust/norm
    y = np.cross(z, [math.cos(p[3]), math.sin(p[3]), 0.])
    ynorm = np.linalg.norm(y)
    if ynorm <= 1e-9:
        raise ValueError('High-level trajectory has singular flatness heading')
    y /= ynorm
    x = np.cross(y, z)
    h = (jerk[:3]-np.dot(jerk[:3], z)*z)/norm
    command[10:] = [-np.dot(h, y), np.dot(h, x), z[2]*v[3]]
    return command


class FlightCommands(StreamingCommands):
    """Single-plant airborne API; disarming holds, emergency suspends the plant.

    The airborne hold is explicit protocol testing, not an unpowered grounded
    drone. Physics, motor dynamics and transport delay remain the native model.
    """
    def __init__(self, robot, evidence=None):
        super().__init__(robot, evidence)
        self.armed = True
        self.acceleration_only = False
        self.emergency_latched = False
        self.group_mask = 0
        self.high_level_enabled = True
        self.mode = 'airborne_hold'
        self.disarmed_s = None          # simulation time the last disarm took effect
        self.plan = self.future_plan = None
        self.waiting_plan = None
        self.release_s = None
        self.stream_cutoff_s = None
        self.late_stream_discarded = 0
        self.api_received = self.api_scheduled = self.generated = 0

    def reset(self):
        super().reset()
        with self.lock:
            self.armed, self.emergency_latched = True, False
            self.acceleration_only = False
            self.mode = 'airborne_hold'
            self.disarmed_s = None
            self.plan = self.future_plan = self.release_s = None
            self.waiting_plan = None
            self.stream_cutoff_s = None

    def status(self):
        with self.lock:
            return dict(armed=self.armed, acceleration_only=self.acceleration_only, emergency_latched=self.emergency_latched,
                flight_mode=self.mode, api_received=self.api_received, api_scheduled=self.api_scheduled,
                high_level_samples=self.generated, group_mask=self.group_mask,
                late_stream_discarded=self.late_stream_discarded,
                high_level_waiting_for_notify=self.waiting_plan is not None)

    def submit(self, packet, snapshot):
        if not self.armed or self.emergency_latched:
            raise ValueError('Streaming requires an armed, non-emergency simulation')
        if self.acceleration_only and any(vector(packet.get('command'), 13)[10:]):
            raise ValueError('Acceleration-only mode requires zero body-rate feedforward')
        return super().submit(packet, snapshot)

    def submit_api(self, packet, snapshot):
        action = packet.get('action')
        if packet.get('robot') != self.robot or action not in ('plan', 'arm', 'notify', 'group_mask', 'high_level'):
            raise ValueError('Unsupported flight API action or robot')
        if snapshot.error or self.emergency_latched:
            raise ValueError('Simulation has an error or latched emergency')
        payload = packet.get('payload', {})
        if action == 'plan':
            if self.acceleration_only:
                raise ValueError('High-level trajectories require PVA mode')
            payload = validate_plan(payload)
            if not self.armed or not self.high_level_enabled:
                raise ValueError('High-level commands require armed simulation and enabled commander')
        elif action in ('arm', 'high_level'):
            if type(payload.get('enabled')) is not bool:
                raise ValueError('Arm/high-level enabled must be boolean')
            if action == 'arm' and not payload['enabled']:
                payload = dict(payload, hold=vector(payload.get('hold'), 13))
        elif action in ('notify', 'group_mask'):
            key, upper = ('millis', 60000) if action == 'notify' else ('mask', 255)
            if type(payload.get(key)) is not int or not 0 <= payload[key] <= upper:
                raise ValueError(f'{key} must be an integer in 0..{upper}')
            if action == 'notify':
                cutoff = payload.get('cutoff_stamp_s', snapshot.state.time_s)
                if type(cutoff) not in (int, float) or not math.isfinite(cutoff):
                    raise ValueError('Notify streaming cutoff must be finite')
                # ROS services and topics have independent delivery order. All
                # stream stamps valid at notification receipt belong to the
                # closing stream, including the bridge's +20-ms clock allowance.
                payload = dict(payload, cutoff_stamp_s=max(float(cutoff), snapshot.state.time_s+.02))
        with self.lock:
            if len(self.pending) >= 128:
                raise ValueError('Flight command queue is full')
            receipt = dict(api_action=action, payload=payload, receipt_sim_time_s=snapshot.state.time_s,
                           generation=self.generation)
            self.pending.append(receipt)
            self.api_received += 1
        if self.evidence:
            self.evidence.write('api_received', **receipt)
        return receipt

    def apply_parameters(self, packet, snapshot):
        """Commit a validated parameter batch under one lock, before its ACK.

        Parameters cannot be queued: disarm clears pending flight commands, and
        an accepted ROS parameter transaction must not disappear with them.
        """
        if packet.get('robot') != self.robot:
            raise ValueError('Parameter robot mismatch')
        updates = packet.get('payload')
        if not isinstance(updates, dict) or not updates or set(updates) - {'high_level', 'group_mask'}:
            raise ValueError('Invalid parameter batch')
        for key, value in updates.items():
            upper = 1 if key == 'high_level' else 255
            if type(value) is not int or not 0 <= value <= upper:
                raise ValueError(f'{key} must be an integer in 0..{upper}')
        with self.lock:
            if snapshot.error or self.emergency_latched:
                raise ValueError('Simulation has an error or latched emergency')
            if 'high_level' in updates:
                self.high_level_enabled = bool(updates['high_level'])
                if not self.high_level_enabled:
                    self.plan = self.future_plan = self.waiting_plan = None
                    self.pending = type(self.pending)(item for item in self.pending
                        if item.get('api_action') != 'plan')
            if 'group_mask' in updates:
                self.group_mask = updates['group_mask']
            receipt = dict(receipt_sim_time_s=snapshot.state.time_s,
                           generation=self.generation, parameters=dict(updates))
        return receipt

    def halt(self):
        with self.lock:
            self.pending.clear()
            self.plan = self.future_plan = None
            self.waiting_plan = None
            self.emergency_latched, self.armed = True, False
            self.mode = 'emergency'

    def events(self, start_s, end_s):
        events = []
        with self.lock:
            receipt = self.pending.popleft() if self.pending else None
            if receipt is not None and 'api_action' not in receipt:
                self.scheduled += 1
                late = self.stream_cutoff_s is not None and receipt['stamp_s'] <= self.stream_cutoff_s+1e-9
                if late and self.mode == 'high_level':
                    self.late_stream_discarded += 1
                    receipt['discard_reason'] = 'pre-notify stream packet arrived after high-level activation'
                else:
                    if not late:
                        self.plan = self.future_plan = self.release_s = self.stream_cutoff_s = None
                    self.mode = 'streaming'
                    events.append((start_s, np.asarray(receipt['command'], dtype=np.float64)))
            elif receipt is not None:
                self.api_scheduled += 1
                action, payload = receipt['api_action'], receipt['payload']
                if action == 'notify':
                    self.release_s = start_s+payload['millis']/1000.
                    self.stream_cutoff_s = payload['cutoff_stamp_s']
                    if self.waiting_plan is not None:
                        self.future_plan = (self.release_s, self.waiting_plan)
                        self.waiting_plan = None
                elif action == 'plan':
                    # Separate ROS service clients can deliver land before
                    # notify, even when the script calls notify first. Retain
                    # the plan at lower priority until explicit stream release.
                    if self.mode == 'streaming' and self.release_s is None:
                        self.waiting_plan = payload
                    else:
                        activation = max(start_s, self.release_s or start_s)
                        self.future_plan = (activation, payload)
                elif action == 'arm':
                    self.armed = payload['enabled']
                    if not self.armed:
                        self.plan = self.future_plan = self.release_s = None
                        self.waiting_plan = None
                        self.mode = 'disarmed_airborne_hold'
                        self.disarmed_s = start_s
                        self.pending.clear()
                        events.append((start_s, np.asarray(payload['hold'], dtype=np.float64)))
                elif action == 'group_mask':
                    self.group_mask = payload['mask']
                elif action == 'high_level':
                    self.high_level_enabled = payload['enabled']
                    if not self.high_level_enabled:
                        self.plan = self.future_plan = None
                        self.waiting_plan = None
            if self.future_plan is not None and start_s+1e-10 >= self.future_plan[0]:
                self.plan, self.future_plan = self.future_plan, None
                self.mode, self.release_s = 'high_level', None
            if self.plan is not None and self.armed:
                activation, plan = self.plan
                events.append((start_s, polynomial_reference(plan, start_s-activation)))
                self.generated += 1
        if receipt is not None and self.evidence:
            self.evidence.write('api_scheduled' if 'api_action' in receipt else 'command_scheduled',
                                **receipt, requested_time_s=start_s)
        return events
