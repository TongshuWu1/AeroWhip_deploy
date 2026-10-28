"""Host ownership, bounded streaming transport and retained connection evidence."""
from collections import deque
from dataclasses import replace
import errno
import json
from pathlib import Path
import socket
import subprocess
from threading import Event, Lock, Thread
from time import monotonic

import numpy as np

from aerowhip.deployment.ros_bridge import PacketReader, encode, vector
from aerowhip.viewer.display_state import simulation_frame
from aerowhip.viewer.sources import SimulationSource


class Evidence:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.lock = Lock()
        self.file = (self.directory/'connection.jsonl').open('x', encoding='utf-8')

    def write(self, kind, **values):
        with self.lock:
            self.file.write(json.dumps(dict(kind=kind, wall_monotonic_s=monotonic(), **values), allow_nan=False)+'\n')

    def close(self):
        with self.lock:
            self.file.close()


class StreamingCommands:
    """FIFO receipt commands: one event at the next available output boundary.

    Header stamps are provenance/validity checks, not backdated plant events.
    Bounded bursts remain ordered; excess packets fail visibly rather than drop.
    """
    def __init__(self, robot, evidence=None):
        self.robot, self.evidence = robot, evidence
        self.lock = Lock()
        self.pending = deque()
        self.received = self.scheduled = self.generation = 0
        self.previous_sequence = -1
        self.previous_stamp = None

    def reset(self):
        with self.lock:
            self.pending.clear()
            self.previous_stamp = None
            self.generation += 1

    def counts(self):
        with self.lock:
            return self.received, self.scheduled, self.generation

    def submit(self, packet, snapshot):
        if packet.get('robot') != self.robot or packet.get('frame_id') != 'world':
            raise ValueError('Command robot/frame mismatch')
        sequence, stamp = packet.get('sequence'), packet.get('stamp_s')
        command = vector(packet.get('command'), 13)
        if type(sequence) is not int or type(stamp) not in (int, float) or not np.isfinite(stamp):
            raise ValueError('Command sequence/timestamp is invalid')
        if not snapshot.running or snapshot.error:
            raise ValueError('Simulation is paused or has an error')
        if not snapshot.state.time_s-.5 <= stamp <= snapshot.state.time_s+.02:
            raise ValueError('Command timestamp must use the current simulation clock')
        with self.lock:
            if sequence <= self.previous_sequence or (self.previous_stamp is not None and stamp < self.previous_stamp-1e-9):
                raise ValueError('Command sequence/timestamp regressed')
            if len(self.pending) >= 128:
                raise ValueError('Streaming command queue is full')
            receipt = dict(sequence=sequence, stamp_s=float(stamp), command=command,
                           receipt_sim_time_s=snapshot.state.time_s, generation=self.generation)
            self.pending.append(receipt)
            self.previous_sequence, self.previous_stamp = sequence, stamp
            self.received += 1
        if self.evidence:
            self.evidence.write('command_received', **receipt)
        return receipt

    def events(self, start_s, end_s):
        with self.lock:
            if not self.pending:
                return []
            receipt = self.pending.popleft()
            self.scheduled += 1
        if self.evidence:
            self.evidence.write('command_scheduled', **receipt, requested_time_s=start_s)
        return [(start_s, np.asarray(receipt['command'], dtype=np.float64))]


class RosSimulationSource(SimulationSource):
    has_trajectory = False

    def __init__(self, simulation, commands, session, evidence=None):
        self.commands, self.session, self.evidence = commands, session, evidence
        self.bridge_error = None
        self._recorded_key = None
        self._packet = None
        self.display_epoch = 0
        if hasattr(commands, 'acceleration_only'):
            # An armed acceleration-only flight script runs the host law itself: its acceleration
            # goes straight to the firmware; otherwise the simulated host tracks its P/V/A reference.
            simulation.dynamics.controller.direct = lambda: commands.acceleration_only and commands.armed
        super().__init__(simulation)

    def _publish(self):
        # Called only during construction or by the native owner thread.
        super()._publish()
        sim = self._simulation
        snapshot = self.latest
        frame = replace(simulation_frame(snapshot.state, name=self.commands.robot),
                        source_label='Simulation flight · ROS')
        frame = replace(frame, robots=tuple(replace(robot, trajectory_epoch=self.display_epoch)
                                           for robot in frame.robots))
        snapshot = replace(snapshot, frame=frame, error=snapshot.error or self.bridge_error)
        state, runtime = snapshot.state, sim.dynamics
        received, scheduled, generation = self.commands.counts()
        packet = dict(kind='state', session=self.session, robot=self.commands.robot,
            frame_id='world', time_s=state.time_s, running=snapshot.running, error=snapshot.error,
            step_count=snapshot.step_count, generation=generation,
            command_delay_s=runtime.delay_s, controller_period_s=runtime.controller_period_s,
            output_period_s=sim.timestep_s, commands_received=received, commands_scheduled=scheduled,
            position_world_m=state.drone.position_world_m.tolist(),
            quaternion_xyzw=state.drone.quaternion_body_to_world_xyzw.tolist(),
            velocity_world_mps=state.drone.linear_velocity_world_mps.tolist(),
            omega_body_radps=state.drone.angular_velocity_body_radps.tolist(),
            acceleration_world_mps2=state.drone.linear_acceleration_world_mps2.tolist(),
            cable_positions_world_m=state.cable_node_positions_world_m.tolist(),
            cable_velocities_world_mps=state.cable_node_velocities_world_mps.tolist(),
            marker_indices=state.cable_marker_indices.tolist(),
            requested_command=runtime.requested_command.tolist(),
            applied_command=runtime.applied_command.tolist(),
            requested_events=[(t,c.tolist()) for t,c in runtime.last_requested_events],
            applied_events=[(t,c.tolist()) for t,c in runtime.last_applied_events],
            controller_update_times=list(runtime.last_controller_update_times))
        if hasattr(self.commands, 'status'):
            packet.update(self.commands.status())
        with self._lock:
            self._latest, self._packet = snapshot, packet
        key = generation, snapshot.step_count
        if self.evidence and key != self._recorded_key:
            self.evidence.write(**packet)
            self._recorded_key = key

    def _execute(self, action, value):
        if self.bridge_error and (action in ('toggle', 'step') or (action == 'running' and value)):
            raise RuntimeError(self.bridge_error)
        if action == 'acceleration_mode':
            runtime = self._simulation.dynamics
            if type(value) is not bool or not hasattr(self.commands, 'acceleration_only'):
                raise ValueError('Invalid acceleration-mode request')
            with self.commands.lock:
                # Settled: disarmed, no queued commands, and the disarm's hold through the transport
                # delay. (Not requested == applied: the host law recomputes the command every step,
                # so the delayed applied command never equals the newest request exactly.)
                disarmed = self.commands.disarmed_s
                if (self.commands.armed or self.commands.pending or
                        self.commands.mode != 'disarmed_airborne_hold' or disarmed is None or
                        self._simulation.state.time_s < disarmed+runtime.delay_s+runtime.controller_period_s):
                    raise ValueError('Disarm and let pending commands settle before changing controller mode')
                self.commands.acceleration_only = value
            return dict(receipt_sim_time_s=self._simulation.state.time_s, acceleration_only=value)
        if action == 'checkpoint':
            checkpoint = super()._execute(action, value)
            if getattr(self.commands, 'acceleration_only', False):
                # Generic PVA restore must not silently resume this with PVA control.
                checkpoint['schema'] = 'aerowhip_ros_acceleration_runtime_v1'
                checkpoint['acceleration_only'] = True
            return checkpoint
        if action == 'emergency':
            self.commands.halt()
            self._simulation.running = False
            self.bridge_error = 'Emergency stop latched; reset or restart the simulation'
        elif action == 'reset' and self.bridge_error and self.bridge_error.startswith('Emergency stop'):
            self.bridge_error = None
            return super()._execute(action, value)
        elif action == 'connection':
            self.bridge_error = value
            if value:
                self._simulation.running = False
                self.commands.reset()
        else:
            return super()._execute(action, value)

    def connection_error(self, message):
        self._call('connection', message)

    @property
    def packet(self):
        with self._lock:
            return self._packet

    def close(self):
        if self._thread.is_alive() and not self._stop.is_set() and self.evidence:
            self.running = False
            checkpoint = self.checkpoint()
            np.savez_compressed(self.evidence.directory/'final-checkpoint.npz', state=checkpoint['state'],
                                schema=checkpoint['schema'], acceleration_only=getattr(self.commands, 'acceleration_only', False))
            self.evidence.write('final_checkpoint', time_s=self.state.time_s,
                                step_count=self.step_count, generation=self.commands.counts()[2])
        super().close()


class PortInUseError(RuntimeError):
    """The requested bridge port cannot be bound; no existing process is stopped."""


class DomainInUseError(RuntimeError):
    """Two custom servers cannot share the global Crazyswarm service names."""


class DomainLease:
    def __init__(self, domain):
        import fcntl
        self.file = Path(f'/tmp/aerowhip-simulation-domain-{domain}.lock').open('a')
        try:
            fcntl.flock(self.file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.file.close()
            raise DomainInUseError(f'ROS domain {domain} already has a custom simulation. Close it or select another --domain; --port 0 does not isolate ROS service names.') from None

    def close(self):
        self.file.close()


class SimulationTransport:
    def __init__(self, source, *, port, domain, evidence=None):
        self.source, self.domain, self.evidence = source, domain, evidence
        self.stop, self.connected = Event(), Event()
        self.listener = socket.socket()
        try:
            # Allow a new listener after our accepted connection enters TIME_WAIT.
            # SO_REUSEPORT is deliberately absent: a live listener stays exclusive.
            self.listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.listener.bind(('127.0.0.1', port))
            self.listener.listen(1)
            self.listener.settimeout(.1)
        except Exception as exc:
            self.listener.close()
            if isinstance(exc, OSError) and exc.errno == errno.EADDRINUSE:
                raise PortInUseError(
                    f'Localhost port {port} is unavailable. Close any existing custom simulation first. '
                    'If no simulation is running, use --port 0 to select a free port; '
                    'an older connection may still be waiting for TCP cleanup.') from None
            raise
        self.port = self.listener.getsockname()[1]
        self.peer = None
        self.error = None
        self.thread = Thread(target=self._run, name='AeroWhip ROS transport', daemon=True)
        self.thread.start()

    def _run(self):
        reader = PacketReader()
        last_revision, last_send = -1, 0.
        last_receive = monotonic()
        try:
            while not self.stop.is_set():
                if self.peer is None:
                    try:
                        self.peer, _ = self.listener.accept()
                        self.peer.settimeout(.005)
                    except socket.timeout:
                        continue
                try:
                    data = self.peer.recv(65536)
                except socket.timeout:
                    data = None
                if data == b'':
                    raise ConnectionError('ROS bridge disconnected; restart the custom simulation')
                if data:
                    last_receive = monotonic()
                for packet in reader.feed(data or b''):
                    if not self.connected.is_set():
                        if (packet.get('kind') != 'hello' or packet.get('robot') != self.source.commands.robot or
                                packet.get('domain') != self.domain or packet.get('session') != self.source.session):
                            raise ValueError('ROS bridge identity/domain/session mismatch')
                        self.connected.set()
                        if self.evidence:
                            self.evidence.write('bridge_connected', domain=self.domain, session=self.source.session)
                    elif packet.get('kind') == 'ping':
                        pass
                    elif packet.get('kind') == 'display_session':
                        epoch = packet.get('epoch')
                        if (packet.get('robot') != self.source.commands.robot or type(epoch) is not int
                                or epoch < self.source.display_epoch):
                            raise ValueError('Invalid flight display session')
                        self.source.display_epoch = epoch
                        if self.evidence:
                            self.evidence.write('flight_display_reset', epoch=epoch)
                    elif packet.get('kind') == 'command':
                        try:
                            receipt = self.source.commands.submit(packet, self.source.latest)
                            ack = dict(kind='ack', sequence=packet.get('sequence'), accepted=True,
                                       receipt_sim_time_s=receipt['receipt_sim_time_s'])
                        except (ValueError, TypeError) as exc:
                            ack = dict(kind='ack', sequence=packet.get('sequence'), accepted=False, error=str(exc))
                            if self.evidence:
                                self.evidence.write('command_rejected', packet=packet, error=str(exc))
                        self.peer.sendall(encode(ack))
                    elif packet.get('kind') == 'api':
                        try:
                            if packet.get('action') == 'emergency':
                                if packet.get('robot') != self.source.commands.robot:
                                    raise ValueError('Emergency robot mismatch')
                                self.source._call('emergency')
                                receipt = dict(receipt_sim_time_s=self.source.state.time_s)
                                if self.evidence:
                                    self.evidence.write('emergency', **receipt)
                            elif packet.get('action') == 'parameters':
                                if packet.get('payload', {}).keys() == {'acceleration_only'}:
                                    if packet.get('robot') != self.source.commands.robot:
                                        raise ValueError('Parameter robot mismatch')
                                    receipt = self.source._call('acceleration_mode', packet['payload']['acceleration_only'])
                                else:
                                    receipt = self.source.commands.apply_parameters(packet, self.source.latest)
                                if self.evidence:
                                    self.evidence.write('parameters_committed', **receipt)
                            else:
                                receipt = self.source.commands.submit_api(packet, self.source.latest)
                            ack = dict(kind='api_ack', request_id=packet.get('request_id'), accepted=True,
                                       receipt_sim_time_s=receipt['receipt_sim_time_s'])
                        except (ValueError, TypeError, RuntimeError) as exc:
                            ack = dict(kind='api_ack', request_id=packet.get('request_id'), accepted=False, error=str(exc))
                            if self.evidence:
                                self.evidence.write('api_rejected', packet=packet, error=str(exc))
                        self.peer.sendall(encode(ack))
                    else:
                        raise ValueError('Unsupported bridge packet')
                snapshot = self.source.latest
                if self.connected.is_set() and monotonic()-last_receive > 2.:
                    raise ConnectionError('ROS bridge stopped responding; restart the custom simulation')
                if self.connected.is_set() and (snapshot.revision != last_revision or monotonic()-last_send > .1):
                    self.peer.sendall(encode(self.source.packet))
                    last_revision, last_send = snapshot.revision, monotonic()
            if self.peer is not None:
                self.peer.sendall(encode(dict(kind='shutdown')))
        except (OSError, ValueError) as exc:
            if not self.stop.is_set():
                self.error = str(exc)
                try:
                    self.source.connection_error(self.error)
                except RuntimeError:
                    pass
                if self.evidence:
                    self.evidence.write('bridge_error', error=self.error)
        finally:
            if self.peer is not None:
                self.peer.close()

    def close(self):
        self.stop.set()
        self.thread.join(5.)
        self.listener.close()
        if self.thread.is_alive():
            raise RuntimeError('ROS transport did not stop within 5 s')


def launch_container_script(container, script, arguments, *, domain, log=None, start_new_session=False):
    """Run exact checked host source in ROS Python, with no image/mount edits."""
    command = ['docker', 'exec', '-i', '-e', f'ROS_DOMAIN_ID={domain}', container,
               'bash', '-lc', 'source /opt/ros/jazzy/setup.bash; source /opt/crazyswarm2_ws/install/setup.bash; exec python3 -u - "$@"',
               'aerowhip-custom-simulation', *map(str, arguments)]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=log,
                               stderr=subprocess.STDOUT, start_new_session=start_new_session)
    try:
        process.stdin.write(Path(script).read_bytes())
        process.stdin.close()
    except Exception:
        process.terminate()
        process.wait(5.)
        raise
    return process
