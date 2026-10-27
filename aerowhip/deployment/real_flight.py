"""Measured OptiTrack display with optional prescribed-motion cable simulation."""
import json
from pathlib import Path
import re
import socket
import subprocess
from threading import Event, RLock, Thread
from time import perf_counter

import numpy as np

from aerowhip.deployment.cable_filter import cable_candidates
from aerowhip.deployment.observer import rotation_log
from aerowhip.deployment.ros_bridge import vector
from aerowhip.deployment.ros_real_bridge import PacketReader, cable_candidate_mask
from aerowhip.physics.state import DroneState, quaternion_xyzw_to_rotation_matrix
from aerowhip.viewer.display_state import CableDisplay, CableTrackingDisplay, DisplayFrame, Provenance, RobotDisplay
from aerowhip.viewer.sources import DisplayFeed


def select_real_robots(base, selected=None):
    if not isinstance(base, dict) or not isinstance(base.get('robots'), dict):
        raise ValueError('Crazyflies YAML must contain a robots mapping')
    types = base.get('robot_types', {})
    robots = []
    for name, data in base['robots'].items():
        robot_type = types.get(data.get('type'), {})
        if data.get('enabled') is not True or robot_type.get('connection', 'crazyflie') != 'crazyflie':
            continue
        if selected is not None and name not in selected:
            continue
        if not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', name):
            raise ValueError(f'Invalid ROS robot name: {name!r}')
        tracking = robot_type.get('motion_capture', {}).get('tracking')
        if tracking not in ('vendor', 'librigidbodytracker'):
            raise ValueError(f'{name} has no enabled motion-capture pose source')
        robots.append(dict(name=name, type=data.get('type'),
            initial_position=vector(data.get('initial_position'), 3), tracking=tracking,
            provenance='Measured' if tracking == 'vendor' else 'Estimated'))
    if not robots or (selected is not None and set(selected) != {r['name'] for r in robots}):
        raise ValueError('Select enabled Crazyflies with motion capture; no matching robots found')
    return robots


def fresh(packet, name, key, max_age_s=.5):
    """A command echo from the relay packet if it is recent (the flight script sends at 50 Hz), else None."""
    command = packet.get('commands', {}).get(name, {})
    if key not in command:
        return None
    age = packet.get('ros_time_s', 0.)-command.get(key+'_time_s', -1e9)
    return np.asarray(command[key], float) if age <= max_age_s else None


class PoseDerivatives:
    """Velocity, body rate and acceleration of a tracked body from its recent OptiTrack poses, for
    display: endpoint derivatives of least-squares quadratics over fixed time spans (velocity and
    body rate over `rate_span_s`, acceleration over `acceleration_span_s`), so the smoothing does
    not depend on the capture rate. Differentiation amplifies pose noise, the second derivative
    most, hence the longer acceleration span. The policy's estimator (deployment.observer) is
    separate and unchanged."""

    def __init__(self, rate_span_s=.08, acceleration_span_s=.15):
        self.rate_span, self.acceleration_span = rate_span_s, acceleration_span_s
        self.samples = []

    @staticmethod
    def _fit(t, values):
        u = t-t[-1]
        design = np.column_stack((np.ones_like(u), u, u*u))
        return np.linalg.lstsq(design, values-values[-1], rcond=None)[0]

    def update(self, time_s, p, q):
        self.samples.append((float(time_s), np.asarray(p, float), quaternion_xyzw_to_rotation_matrix(np.asarray(q, float))))
        newest = self.samples[-1][0]
        while self.samples and newest-self.samples[0][0] > self.acceleration_span:
            self.samples.pop(0)
        t = np.array([x[0] for x in self.samples])
        recent = newest-t <= self.rate_span
        if recent.sum() < 3 or t[-1]-t[recent][0] < 1e-3:
            return None, None, None
        positions = np.array([x[1] for x in self.samples])
        velocity = self._fit(t[recent], positions[recent])[1]
        R = self.samples[-1][2]
        logs = np.array([rotation_log(R.T@x[2]) for x, keep in zip(self.samples, recent) if keep])
        omega = self._fit(t[recent], logs)[1]
        acceleration = 2*self._fit(t, positions)[2] if len(t) >= 5 else None
        return velocity, omega, acceleration


class RealFlightFeed(DisplayFeed):
    def __init__(self, configuration, robots, frame_id, session, *, cable_model=None, tracking_settings=None, capture_only=False):
        self.robots = {r['name']: r for r in robots}
        self.frame_id, self.session = frame_id, session
        self.source_label = 'Hand capture · OptiTrack' if capture_only else 'Real flight · OptiTrack'
        self.sequence = -1
        self.generation = 0
        self.transport = None
        self.cable_model = cable_model
        self._record_lock = RLock()
        self._recorder = None
        self.recording_directory = None
        self.recording_metadata = None
        self.last_recording = None
        self.last_recording_error = None
        self.tracker = None
        self.tracking_display = None
        self.tracking_generation = 0
        self.tracking_status = 'awaiting point cloud'
        self.reconstruction = None
        self.derivatives = {}      # per robot: PoseDerivatives for the displayed velocity/acceleration
        self.tracked = set()       # robots that have had an OptiTrack pose in this session
        if tracking_settings is not None:
            # The online labeller: the same live CableFilter that labels the recordings and runs in
            # the policy's estimator (deployment.observer), on the cloud the relay forwards.
            from aerowhip.deployment.cable_filter import CableFilter
            if len(self.robots) != 1:
                raise ValueError('Cable tracking requires one selected robot (--robot)')
            self.tracker = CableFilter(np.asarray(tracking_settings.marker_distances_m, float))
        if cable_model is not None:
            from aerowhip.physics.live_cable import LiveCable
            if (not np.allclose(configuration.platform.attachment_offset_body_m,
                    cable_model.attachment_tracking, atol=1e-12, rtol=0)
                    or not np.array_equal(configuration.platform.cable_segment_lengths_m,
                        cable_model.config['cable']['marker_interval_lengths_m'])):
                raise ValueError('Viewer attachment/cable lengths must match the selected cable model')
            self.cables = {name: LiveCable(cable_model) for name in self.robots}
        else:
            self.cables = {}
        super().__init__(DisplayFrame(0., (), frame_id=frame_id,
            source_label=self.source_label+' · awaiting capture'), configuration)

    def receive(self, packet):
        with self._record_lock:
            recorder = self._recorder
            if recorder is not None:
                recorder.event('capture', packet=packet)
            self._receive(packet)
            if recorder is not None:
                if packet.get('kind') == 'markers':
                    recorder.event('cable_reconstruction', result=self.reconstruction)
                elif packet.get('kind') == 'state':
                    recorder.frame(self.latest)

    def configure_recording(self, directory, metadata):
        with self._record_lock:
            self.recording_directory = Path(directory)
            self.recording_metadata = metadata

    def start_recording(self):
        from aerowhip.viewer.recording import DisplayRecorder
        with self._record_lock:
            if self.recording_directory is None:
                raise RuntimeError('Recording output is not configured')
            if self._recorder is not None:
                raise RuntimeError('Stop the current recording before starting another')
            self._recorder = DisplayRecorder(self.recording_directory, self.configuration,
                self.recording_metadata, self.latest)
            self.last_recording_error = None
            return self._recorder.manifest

    def stop_recording(self):
        with self._record_lock:
            if self._recorder is None:
                return self.last_recording
            self.last_recording = self._recorder.close()
            self.last_recording_error = self._recorder.error
            self._recorder = None
            return self.last_recording

    @property
    def recording_status(self):
        with self._record_lock:
            recorder = self._recorder
            return dict(active=recorder is not None, writing=bool(recorder and recorder.active),
                duration_s=0. if recorder is None else perf_counter()-recorder.started,
                frames=0 if recorder is None else recorder.frame_count,
                path=self.last_recording if recorder is None else recorder.manifest,
                error=self.last_recording_error if recorder is None else recorder.error)

    def report_transport_error(self, error):
        with self._record_lock:
            self.publish(self.latest.frame, error=error)
            if self._recorder is not None:
                self._recorder.event('transport_error', error=error)
                self._recorder.frame(self.latest)

    def _receive(self, packet):
        if packet.get('session') != self.session or packet.get('frame_id') != self.frame_id:
            raise ValueError('Real-view session/frame mismatch')
        if packet.get('kind') == 'ready':
            return
        if packet.get('kind') == 'markers':
            self.receive_markers(packet)
            return
        if packet.get('kind') != 'state' or type(packet.get('sequence')) is not int or packet['sequence'] <= self.sequence:
            raise ValueError('Unexpected/regressing real-view packet')
        time_s = float(packet['time_s'])
        if not np.isfinite(time_s):
            raise ValueError('Invalid display timeline')
        generation = packet.get('generation')
        if type(generation) is not int or generation < self.generation:
            raise ValueError('Invalid/regressing capture generation')
        if generation == self.generation and time_s < self.latest.frame.time_s:
            raise ValueError('Display clock regressed without a generation change')
        if generation != self.generation:
            for cable in self.cables.values():
                cable.reset()
            self.derivatives.clear()
            if self.tracker is not None and generation != self.tracking_generation:
                self.tracker.reset()
                self.tracking_display = None
                self.tracking_generation = generation
        if packet.get('pose_samples_complete', True) is not True:
            for cable in self.cables.values():
                cable.fail('Cable capture sample buffer overflowed; restart the real viewer')
        # Batched samples were verified in the ROS capture callback. Complete
        # that accepted prefix even if heartbeat verification changes afterward.
        samples = packet.get('pose_samples')
        if samples is None:
            samples = packet.get('robots', []) if packet.get('capture_type') in (
                'optitrack', 'optitrack_closed_source') else []
        for data in samples:
            if data['name'] in self.cables:
                self.cables[data['name']].observe(float(data['time_s']),
                    vector(data['position_world_m'], 3), vector(data['quaternion_xyzw'], 4))
        displays = []
        for data in packet.get('robots', []):
            name = data['name']
            if name not in self.robots:
                raise ValueError('Unconfigured robot in real-view packet')
            p = np.asarray(vector(data['position_world_m'], 3))
            q = np.asarray(vector(data['quaternion_xyzw'], 4))
            velocity, omega, acceleration = self.derivatives.setdefault(name, PoseDerivatives()).update(data['time_s'], p, q)
            attachment = p+quaternion_xyzw_to_rotation_matrix(q)@self.configuration.platform.attachment_offset_body_m
            simulated = self.cables.get(name)
            cable = None if simulated is None or simulated.latest is None else CableDisplay(
                simulated.latest.time_s, Provenance.SIMULATED,
                simulated.latest.positions_world_m, self.cable_model.marker_indices)
            displays.append(RobotDisplay(name, float(data['time_s']),
                Provenance(self.robots[name]['provenance']), DroneState(p, q, velocity, omega, acceleration), attachment,
                cable=cable,
                cable_tracking=(self.tracking_display if self.tracking_display is not None
                    and self.tracking_display.time_s <= time_s else None),
                trajectory_epoch=packet.get('flight_epochs', {}).get(name, 0),
                commanded_acceleration_world_mps2=fresh(packet, name, 'acceleration_mps2'),
                desired_acceleration_world_mps2=fresh(packet, name, 'desired_mps2')))
        label = self.source_label
        if self.cables:
            label += ' · simulated cable'
        if self.tracker is not None:
            label += ' · cable estimate: '+self.tracking_status
            if self.tracking_display is not None and time_s-self.tracking_display.time_s > .25:
                label += ' (stale)'
        if packet.get('capture_type') is None:
            label += ' · awaiting capture source'
        for name in packet.get('robots', []):
            self.tracked.add(name['name'])
        missing = packet.get('missing_robots') or []
        never = [n for n in missing if n not in self.tracked]
        lost = [n for n in missing if n in self.tracked]
        if never:
            label += ' · waiting for the first OptiTrack pose of '+', '.join(never)+' (the stack starts once the drone answers on the radio)'
        if lost:
            label += ' · OptiTrack lost '+', '.join(lost)
        for name, status in packet.get('status', {}).items():
            armed = bool(status.get('supervisor_info', 0) & 2)
            label += (f" · {name} {status['battery_v']:.2f} V{' armed' if armed else ''}"
                      + (' (stale)' if status.get('age_s', 0.) > 2.5 else ''))
        for name, motors in packet.get('motors', {}).items():
            if motors.get('age_s', 0.) < 1.:
                label += ' · PWM '+' '.join(f'{v/1000:.1f}k' for v in motors['pwm'])
        for name, command in packet.get('commands', {}).items():
            if command.get('age_s', 0.) < 1. and 'acceleration_mps2' in command:
                a = command['acceleration_mps2']
                label += f' · cmd a [{a[0]:+.2f} {a[1]:+.2f} {a[2]:+.2f}] m/s²'
        issues = packet.get('issues', {})
        errors = [f'{name}: {value}' for name, value in issues.items()]
        errors.extend(f'{name}: {cable.error}' for name, cable in self.cables.items() if cable.error)
        error = '; '.join(errors) or None
        frame = DisplayFrame(time_s, tuple(displays), frame_id=self.frame_id, source_label=label)
        self.publish(frame, error=error)
        self.sequence, self.generation = packet['sequence'], generation

    def receive_markers(self, packet):
        if self.tracker is None:
            raise ValueError('Unexpected cable cloud without tracking enabled')
        self.reconstruction = None
        generation = packet['generation']
        if generation < self.tracking_generation:
            self.reconstruction = dict(status='late cloud from an old generation', stamp_ns=packet['stamp_ns'])
            return
        if generation != self.tracking_generation:
            self.tracker.reset()
            self.tracking_display = None
            self.tracking_generation = generation
        matching = [r for r in packet['robots'] if r['name'] in self.robots]
        if len(matching) != 1:
            self.tracking_status = 'cloud has no matching drone pose'
            self.tracking_display = None
            self.reconstruction = dict(status=self.tracking_status, stamp_ns=packet['stamp_ns'])
            return
        data = matching[0]
        if data['stamp_ns'] != packet['stamp_ns']:
            raise ValueError('Cable cloud/pose timestamp mismatch')
        drone = np.asarray(vector(data['position_world_m'], 3))
        root = drone+quaternion_xyzw_to_rotation_matrix(vector(data['quaternion_xyzw'], 4))@self.configuration.platform.attachment_offset_body_m
        points = np.asarray(packet['points_world_m'], dtype=float).reshape(-1, 3)
        try:
            eligible = np.asarray(cable_candidate_mask(packet.get('point_metadata'), len(points)), dtype=bool)
            candidates = cable_candidates(points[eligible], drone)      # as offline: drone markers and duplicates removed
            state = self.tracker.observe(float(packet['time_s']), root, candidates)
        except ValueError as exc:
            self.tracking_status = str(exc)
            self.tracking_display = None
            self.reconstruction = dict(status=str(exc), stamp_ns=packet['stamp_ns'])
            return
        observed = state['index'] >= 0
        labelled = np.isfinite(state['positions']).all(axis=1)
        positions = np.where(labelled[:, None], state['positions'], 0.)
        status = (f'{int(observed.sum())}/{len(observed)} seen · quality {state["quality"]:.2f}' if state['initialized']
                  else 'waiting for a hanging cable with most markers visible')
        self.reconstruction = dict(stamp_ns=packet['stamp_ns'], generation=generation, robot=data['name'], time_s=state['time_s'],
            initialized=bool(state['initialized']), index=state['index'].tolist(), confidence=state['confidence'].tolist(),
            positions_world_m=[p.tolist() if ok else None for p, ok in zip(state['positions'], labelled)],
            candidates=int(len(candidates)), excluded_point_indices=np.flatnonzero(~eligible).tolist(),
            labeller='deployment.cable_filter.CableFilter (the policy estimator and offline labeller)')
        self.tracking_status = status
        self.tracking_display = CableTrackingDisplay(float(packet['time_s']), root, positions, observed & labelled,
            np.asarray(state['confidence']), points[np.isfinite(points).all(axis=1)], status)

    def close(self):
        if self.transport:
            self.transport.close()
        self.stop_recording()
        super().close()


class RealFlightTransport:
    def __init__(self, source, evidence, *, port=0):
        self.source, self.evidence = source, evidence
        self.stop, self.connected = Event(), Event()
        self.peer = None
        self.error = None
        self.listener = socket.socket()
        self.listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            self.listener.bind(('127.0.0.1', port))
            self.listener.listen(1)
            self.listener.settimeout(.1)
        except Exception:
            self.listener.close()
            raise
        self.port = self.listener.getsockname()[1]
        self.thread = Thread(target=self.run, name='OptiTrack display', daemon=True)
        source.transport = self
        self.thread.start()

    def run(self):
        reader = PacketReader()
        try:
            while not self.stop.is_set() and self.peer is None:
                try:
                    self.peer, _ = self.listener.accept()
                    self.peer.settimeout(.1)
                except socket.timeout:
                    continue
            while not self.stop.is_set():
                try:
                    data = self.peer.recv(65536)
                except socket.timeout:
                    continue
                if not data:
                    raise ConnectionError('OptiTrack display relay disconnected')
                for packet in reader.feed(data):
                    # Retain the input before association so failed/ambiguous
                    # reconstructions cannot silently remove source evidence.
                    self.evidence.write('capture_display', packet=packet)
                    self.source.receive(packet)
                    if packet['kind'] == 'markers':
                        self.evidence.write('cable_reconstruction', result=self.source.reconstruction)
                    if packet['kind'] == 'ready':
                        self.connected.set()
        except Exception as exc:
            if not self.stop.is_set():
                self.error = str(exc)
                self.source.report_transport_error(self.error)
                self.evidence.write('display_error', error=self.error)
        finally:
            if self.peer:
                self.peer.close()

    def close(self):
        if self.stop.is_set():
            return
        self.stop.set()
        if self.peer:
            try:
                self.peer.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        self.thread.join(5.)
        self.listener.close()
        if self.thread.is_alive():
            raise RuntimeError('OptiTrack display reader did not stop')


def stop_real_stack(container, session, signum=2):
    """Signal only the launch group owned by this session."""
    script = '''import json,os,signal,sys
from pathlib import Path
session,signum=sys.argv[1],int(sys.argv[2])
record=Path('/tmp')/('aerowhip-real-'+session+'.json')
active=False
if record.exists():
    data=json.loads(record.read_text()); pid=data['pid']
    try:
        environment=Path('/proc')/str(pid)/'environ'
        owned=('AEROWHIP_REAL_SESSION='+session).encode() in environment.read_bytes().split(b'\\0')
        if data['session'] != session or not owned or os.getpgid(pid) != pid:
            raise RuntimeError('Refusing to signal an unowned ROS launch')
        os.killpg(pid,signum)
        active=True
    except (ProcessLookupError,FileNotFoundError):
        pass
print(json.dumps(active))
'''
    result = subprocess.run(['docker', 'exec', '-i', container, 'python3', '-c', script,
                    session, str(signum)], check=True, capture_output=True, text=True)
    return json.loads(result.stdout)


def existing_real_nodes(container, domain):
    script = '''import json,time,rclpy
from rclpy.node import Node
rclpy.init(args=[])
n=Node('aerowhip_real_launch_check',enable_rosout=False,start_parameter_services=False)
end=time.monotonic()+1.5
while time.monotonic()<end: rclpy.spin_once(n,timeout_sec=.05)
print(json.dumps(n.get_node_names_and_namespaces()))
n.destroy_node();rclpy.shutdown()
'''
    result = subprocess.run(['docker', 'exec', '-i', '-e', f'ROS_DOMAIN_ID={domain}', container,
        'bash', '-lc', 'source /opt/ros/jazzy/setup.bash; source /opt/crazyswarm2_ws/install/setup.bash; exec python3 -c "$1"',
        'aerowhip-real-check', script], check=True, capture_output=True, text=True)
    return [name for name, namespace in json.loads(result.stdout)
            if name in ('crazyflie_server', 'motion_capture_tracking', 'motion_capture_tracking_node')]
