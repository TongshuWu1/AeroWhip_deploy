"""Crazyswarm ROS command adapter, executable in the existing ROS container.

This file has no host-only dependencies. The launcher sends its exact source to
the container's Python stdin. Physics and visualization remain on the host.
"""
import argparse
import hashlib
import json
import math
import socket
import time

SCHEMA = 'aerowhip_ros_stream_v1'
DEFAULT_PORT = 47654
DEFAULT_DOMAIN = 142
MAX_PACKET_BYTES = 262144


def encode(packet):
    return (json.dumps(dict(packet, schema=SCHEMA), allow_nan=False, separators=(',', ':'))+'\n').encode()


class PacketReader:
    def __init__(self):
        self.buffer = b''

    def feed(self, data):
        self.buffer += data
        result = []
        while b'\n' in self.buffer:
            line, self.buffer = self.buffer.split(b'\n', 1)
            if len(line) > MAX_PACKET_BYTES:
                raise ValueError('Bridge packet exceeds size limit')
            packet = json.loads(line)
            if not isinstance(packet, dict) or packet.get('schema') != SCHEMA:
                raise ValueError('Incompatible bridge schema')
            result.append(packet)
        if len(self.buffer) > MAX_PACKET_BYTES:
            raise ValueError('Bridge packet exceeds size limit')
        return result


def vector(value, size):
    if (not isinstance(value, (list, tuple)) or len(value) != size or
            any(type(x) not in (int, float) or not math.isfinite(x) for x in value)):
        raise ValueError(f'Expected {size} finite numbers')
    return [float(x) for x in value]


def parameter_batch(parameters, robot):
    """Validate the whole ROS transaction before any host-side mutation."""
    updates = {}
    for parameter in parameters:
        name, value = parameter.name, parameter.value
        if name == 'aerowhip.acceleration_only':
            if len(parameters) != 1 or type(value) is not bool:
                raise ValueError('Set acceleration_only alone with a boolean value')
            return {'acceleration_only': value}
        if name == 'use_sim_time':
            if value is not True:
                raise ValueError('Custom simulation requires use_sim_time=true')
            continue
        names = {f'{prefix}.params.{suffix}': key
                 for prefix in ('all', robot)
                 for suffix, key in (('commander.enHighLevel', 'high_level'),
                                     ('hlCommander.groupmask', 'group_mask'))}
        key = names.get(name)
        if key is None:
            raise ValueError('Firmware gains are not emulated; use the explicit simulation model')
        upper = 1 if key == 'high_level' else 255
        if type(value) is not int or not 0 <= value <= upper:
            raise ValueError(f'{name} must be an integer in 0..{upper}')
        if key in updates and updates[key] != value:
            raise ValueError('Conflicting all/robot parameter aliases in one transaction')
        updates[key] = value
    return updates


def full_state_reference(message, frame='world'):
    if message.header.frame_id.lstrip('/') != frame:
        raise ValueError(f'Command frame must be {frame} or /{frame}')
    p, v, a, w = message.pose.position, message.twist.linear, message.acc, message.twist.angular
    q = message.pose.orientation
    x, y, z, qw = vector([q.x, q.y, q.z, q.w], 4)
    if abs(x*x+y*y+z*z+qw*qw-1.) > 1e-6:
        raise ValueError('Command quaternion must be normalized')
    # The native reference contract is PVA/yaw, not a full attitude command.
    if abs(x) > 1e-6 or abs(y) > 1e-6:
        raise ValueError('Command orientation must encode yaw only')
    yaw = math.atan2(2*qw*z, 1-2*z*z)
    return vector([p.x,p.y,p.z,v.x,v.y,v.z,a.x,a.y,a.z,yaw,w.x,w.y,w.z], 13)


def body_velocity(world_velocity, quaternion):
    """nav_msgs/Odometry twist is expressed in its body child frame."""
    vx, vy, vz = vector(world_velocity, 3)
    x, y, z, w = vector(quaternion, 4)
    return [(1-2*(y*y+z*z))*vx+2*(x*y+w*z)*vy+2*(x*z-w*y)*vz,
            2*(x*y-w*z)*vx+(1-2*(x*x+z*z))*vy+2*(y*z+w*x)*vz,
            2*(x*z+w*y)*vx+2*(y*z-w*x)*vy+(1-2*(x*x+y*y))*vz]


def firmware_plan(action, payload, reference):
    """Use the pinned firmware planner, exporting coefficients, not motor outputs."""
    import cffirmware as firm
    duration = payload['duration_s']
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError('Flight command duration must be positive and finite')
    reference = vector(reference, 13)
    planner = firm.planner()
    firm.plan_init(planner)
    if action in ('takeoff', 'land'):
        height = payload['height']
        if not math.isfinite(height):
            raise ValueError('Flight height must be finite')
        function = firm.plan_takeoff if action == 'takeoff' else firm.plan_land
        result = function(planner, firm.mkvec(*reference[:3]), reference[9], height, 0., duration, 0.)
    else:
        current = firm.traj_eval_zero()
        current.pos = firm.mkvec(*reference[:3])
        current.vel = firm.mkvec(*reference[3:6])
        current.acc = firm.mkvec(*reference[6:9])
        current.jerk = firm.vzero()
        current.yaw = reference[9]
        current.omega = firm.mkvec(*reference[10:13])
        goal, yaw = vector(payload['goal'], 3), payload['yaw']
        if not math.isfinite(yaw):
            raise ValueError('Go-to yaw must be finite')
        result = firm.plan_go_to_from(planner, current, payload['relative'], False,
                                     firm.mkvec(*goal), yaw, duration, 0.)
    if result != 0:
        raise ValueError('Firmware planner rejected the command')
    trajectory = planner.trajectory
    pieces = []
    for i in range(trajectory.n_pieces):
        piece = firm.piecewise_get(trajectory, i)
        pieces.append(dict(duration_s=float(piece.duration), coefficients=[
            [float(firm.poly4d_get(piece, axis, k)) for k in range(8)] for axis in range(4)]))
    return dict(pieces=pieces, timescale=1., reverse=False, shift=[0., 0., 0.])


def main():
    import os
    import rclpy
    from builtin_interfaces.msg import Time
    from crazyflie_interfaces.msg import FullState, Status
    from crazyflie_interfaces.srv import Arm, GoTo, Land, NotifySetpointsStop, StartTrajectory, Takeoff, UploadTrajectory
    from rcl_interfaces.msg import ParameterDescriptor, SetParametersResult
    from rclpy.parameter import Parameter
    from rclpy.qos import QoSProfile, DurabilityPolicy
    from std_srvs.srv import Empty
    from ament_index_python.packages import get_package_share_directory
    from pathlib import Path
    from geometry_msgs.msg import Pose, PoseArray, PoseStamped, TransformStamped
    from nav_msgs.msg import Odometry
    from rclpy.clock import Clock as RosClock
    from rclpy.clock import ClockType
    from rclpy.node import Node
    from rosgraph_msgs.msg import Clock
    from std_msgs.msg import String
    from tf2_ros import TransformBroadcaster

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=DEFAULT_PORT)
    parser.add_argument('--robot', default='cf_2')
    parser.add_argument('--session', required=True)
    parser.add_argument('--robot-config', required=True)
    args = parser.parse_args()
    config = json.loads(args.robot_config)
    if config['name'] != args.robot or config['frame'] != 'world':
        parser.error('Robot configuration identity/frame mismatch')
    rclpy.init()
    probe = Node('aerowhip_probe_'+hashlib.sha256(args.session.encode()).hexdigest()[:8])
    try:
        deadline = time.monotonic()+.5
        while time.monotonic() < deadline:
            rclpy.spin_once(probe, timeout_sec=.02)
            if 'crazyflie_server' in probe.get_node_names():
                raise RuntimeError('ROS domain already contains crazyflie_server; close that server or choose a different simulation domain')
    finally:
        probe.destroy_node()

    class Bridge(Node):
        def __init__(self):
            super().__init__('crazyflie_server', parameter_overrides=[Parameter('use_sim_time', value=True)])
            self.reader = PacketReader()
            self.last = None
            self.sequence = self.rejected = self.accepted = 0
            self.done = False
            self.last_receive = time.monotonic()
            self.last_ping = 0.
            self.request_id = self.api_accepted = self.api_rejected = 0
            self.api_acks = {}
            self.trajectories = {}
            self.command_error = None
            self.syncing_parameters = False
            self.flight_publishers = set()
            self.flight_epoch = 0
            self.last_flight_check = -math.inf
            metadata = {'aerowhip.simulation': True, 'aerowhip.session': args.session,
                'aerowhip.airborne_landing_height_m': config['airborne_landing_height_m'],
                f'robots.{args.robot}.enabled': True, f'robots.{args.robot}.uri': config['uri'],
                f'robots.{args.robot}.initial_position': config['initial_position'],
                f'robots.{args.robot}.type': config['type'], f'robots.{args.robot}.reference_frame': config['frame']}
            for key, value in metadata.items():
                self.declare_parameter(key, value, ParameterDescriptor(read_only=True))
            self.declare_parameter('aerowhip.acceleration_only', False)
            for name in ('all', args.robot):
                self.declare_parameter(f'{name}.params.commander.enHighLevel', 1)
                self.declare_parameter(f'{name}.params.hlCommander.groupmask', 0)
            self.peer = None
            deadline = time.monotonic()+10.
            while time.monotonic() < deadline:
                try:
                    self.peer = socket.create_connection(('127.0.0.1', args.port), timeout=.1)
                    break
                except OSError:
                    time.sleep(.05)
            if self.peer is None:
                raise RuntimeError('Host simulation did not accept the ROS bridge within 10 s')
            self.peer.settimeout(.005)
            self.peer.sendall(encode(dict(kind='hello', robot=args.robot,
                domain=int(os.environ.get('ROS_DOMAIN_ID', '0')), session=args.session)))
            self.clock_pub = self.create_publisher(Clock, '/clock', 10)
            self.pose_pub = self.create_publisher(PoseStamped, f'/{args.robot}/pose', 10)
            self.state_pub = self.create_publisher(FullState, f'/{args.robot}/simulation_state', 10)
            self.cable_pub = self.create_publisher(PoseArray, f'/{args.robot}/simulation_cable', 10)
            self.odom_pub = self.create_publisher(Odometry, f'/{args.robot}/odom', 10)
            self.status_pub = self.create_publisher(String, '/aerowhip/simulation/status', 10)
            self.drone_status_pub = self.create_publisher(Status, f'/{args.robot}/status', 10)
            self.tf = TransformBroadcaster(self)
            self.create_subscription(FullState, f'/{args.robot}/cmd_full_state', self.command, 100)
            self.create_subscription(FullState, '/all/cmd_full_state', self.command, 100)
            self.description_pub = self.create_publisher(String, f'/{args.robot}/robot_description',
                QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
            description = (Path(get_package_share_directory('crazyflie_description'))/'urdf/crazyflie_description.urdf').read_text()
            self.description_pub.publish(String(data=description.replace('$NAME', args.robot)))
            self.flight_services = []
            for name in (args.robot, 'all'):
                for kind, service in (('arm', Arm), ('takeoff', Takeoff), ('land', Land), ('go_to', GoTo),
                        ('notify_setpoints_stop', NotifySetpointsStop), ('upload_trajectory', UploadTrajectory),
                        ('start_trajectory', StartTrajectory), ('emergency', Empty)):
                    self.flight_services.append(self.create_service(service, f'/{name}/{kind}',
                        lambda request, response, kind=kind: self.service(kind, request, response)))
            self.add_on_set_parameters_callback(self.parameters)
            self.add_post_set_parameters_callback(self.mirror_parameters)
            self.create_timer(.005, self.poll, clock=RosClock(clock_type=ClockType.STEADY_TIME))
            self.get_logger().info(f'Simulation flight: {args.robot}, ROS domain {os.environ.get("ROS_DOMAIN_ID")}, localhost:{args.port}')

        def command(self, message):
            try:
                if self.last is None or not self.last['running'] or self.last.get('error'):
                    raise ValueError('Simulation is not running')
                command = full_state_reference(message, self.last['frame_id'])
                stamp = message.header.stamp.sec+message.header.stamp.nanosec*1e-9
                if not self.last['time_s']-.5 <= stamp <= self.last['time_s']+.02:
                    raise ValueError('Command timestamp is outside the simulation clock window; use use_sim_time:=true')
                self.sequence += 1
                self.peer.sendall(encode(dict(kind='command', sequence=self.sequence,
                    robot=args.robot, stamp_s=stamp, frame_id=self.last['frame_id'], command=command)))
            except (ValueError, OSError) as exc:
                self.rejected += 1
                self.get_logger().warning(f'Rejected full-state command: {exc}')

        def request(self, action, payload=None):
            if self.done or self.last is None or (self.last.get('error') and action != 'emergency'):
                raise ValueError('Simulation feedback is unavailable or has an error')
            if not self.last['running'] and action not in ('emergency', 'arm'):
                raise ValueError('Simulation is paused; resume before sending a flight command')
            self.request_id += 1
            request_id = self.request_id
            try:
                self.peer.sendall(encode(dict(kind='api', request_id=request_id, robot=args.robot,
                                             action=action, payload=payload or {})))
            except OSError:
                self.done = True
                raise
            deadline = time.monotonic()+1.
            while request_id not in self.api_acks and not self.done and time.monotonic() < deadline:
                self.poll()
                time.sleep(.001)
            ack = self.api_acks.pop(request_id, None)
            if ack is None:
                # The host may have committed before the acknowledgement was
                # lost. Stop this session instead of serving stale ROS values.
                self.done = True
                raise ValueError('Host did not acknowledge flight command within 1 s')
            if not ack['accepted']:
                raise ValueError(ack['error'])
            self.command_error = None

        def parameters(self, parameters):
            if self.syncing_parameters:
                return SetParametersResult(successful=True)
            try:
                updates = parameter_batch(parameters, args.robot)
                if updates:
                    self.request('parameters', updates)
                return SetParametersResult(successful=True)
            except (ValueError, OSError) as exc:
                return SetParametersResult(successful=False, reason=str(exc))

        def mirror_parameters(self, parameters):
            if self.syncing_parameters:
                return
            mirrors = []
            for parameter in parameters:
                if '.params.' in parameter.name:
                    prefix, suffix = parameter.name.split('.params.', 1)
                    other = args.robot if prefix == 'all' else 'all'
                    mirrors.append(Parameter(f'{other}.params.{suffix}', value=parameter.value))
            if mirrors:
                self.syncing_parameters = True
                try:
                    self.set_parameters(mirrors)
                finally:
                    self.syncing_parameters = False

        def service(self, kind, request, response):
            try:
                mask = getattr(request, 'group_mask', 0)
                if mask and self.last is not None and not (mask & self.last.get('group_mask', 0)):
                    return response
                reference = self.last['requested_command'] if self.last is not None else None
                if kind == 'emergency':
                    self.request('emergency')
                elif kind == 'arm':
                    payload = dict(enabled=request.arm)
                    if not request.arm:
                        p, q = self.last['position_world_m'], self.last['quaternion_xyzw']
                        yaw = math.atan2(2*(q[3]*q[2]+q[0]*q[1]), 1-2*(q[1]*q[1]+q[2]*q[2]))
                        payload['hold'] = p+[0.]*6+[yaw]+[0.]*3
                    self.request('arm', payload)
                elif kind == 'notify_setpoints_stop':
                    self.request('notify', dict(millis=request.remain_valid_millisecs,
                                               cutoff_stamp_s=self.last['time_s']))
                elif kind in ('takeoff', 'land', 'go_to'):
                    duration = request.duration.sec+request.duration.nanosec*1e-9
                    payload = dict(duration_s=duration)
                    if kind == 'go_to':
                        payload.update(goal=[request.goal.x,request.goal.y,request.goal.z],
                                       yaw=request.yaw, relative=request.relative)
                    else:
                        payload['height'] = request.height
                    self.request('plan', firmware_plan(kind, payload, reference))
                    self.get_logger().info(f'{kind}: {payload}')
                elif kind == 'upload_trajectory':
                    if not request.pieces or len(request.pieces)+request.piece_offset > 64:
                        raise ValueError('Upload supports 1..64 trajectory pieces')
                    pieces = []
                    for piece in request.pieces:
                        duration = piece.duration.sec+piece.duration.nanosec*1e-9
                        if duration <= 0:
                            raise ValueError('Uploaded piece duration must be positive')
                        pieces.append(dict(duration_s=duration, coefficients=[vector(list(c), 8)
                            for c in (piece.poly_x,piece.poly_y,piece.poly_z,piece.poly_yaw)]))
                    original = self.trajectories.get(request.trajectory_id, [])[:]
                    if request.piece_offset > len(original):
                        raise ValueError('Trajectory piece offset leaves a gap')
                    original[request.piece_offset:request.piece_offset+len(pieces)] = pieces
                    self.trajectories[request.trajectory_id] = original
                    self.get_logger().info(f'Uploaded trajectory {request.trajectory_id}: {len(original)} pieces')
                elif kind == 'start_trajectory':
                    pieces = self.trajectories.get(request.trajectory_id)
                    if pieces is None:
                        raise ValueError('Trajectory ID has not been uploaded')
                    shift = [0.,0.,0.]
                    if request.relative:
                        piece = pieces[-1] if request.reversed else pieces[0]
                        t = piece['duration_s'] if request.reversed else 0.
                        start = [sum(c[k]*t**k for k in range(8)) for c in piece['coefficients'][:3]]
                        shift = [reference[k]-start[k] for k in range(3)]
                    self.request('plan', dict(pieces=pieces, timescale=request.timescale,
                        reverse=request.reversed, shift=shift))
                return response
            except (ValueError, TypeError, KeyError, OSError) as exc:
                # Upstream services have empty responses; expose failures in status
                # and logs, never fabricate a successful flight operation.
                self.command_error = f'{kind}: {exc}'
                self.get_logger().error(self.command_error)
                return response

        def poll(self):
            try:
                if time.monotonic()-self.last_flight_check >= .1:
                    self.last_flight_check = time.monotonic()
                    publishers = {bytes(p.endpoint_gid) for topic in
                        (f'/{args.robot}/cmd_full_state', '/all/cmd_full_state')
                        for p in self.get_publishers_info_by_topic(topic)}
                    if publishers-self.flight_publishers:
                        self.flight_epoch += 1
                        self.peer.sendall(encode(dict(kind='display_session', robot=args.robot,
                                                     epoch=self.flight_epoch)))
                    self.flight_publishers = publishers
                if time.monotonic()-self.last_ping > .1:
                    self.peer.sendall(encode(dict(kind='ping')))
                    self.last_ping = time.monotonic()
                # Bounded work, independent of the simulation owner and Qt.
                for _ in range(16):
                    try:
                        data = self.peer.recv(65536)
                    except socket.timeout:
                        break
                    if not data:
                        raise ConnectionError('Host simulation disconnected')
                    self.last_receive = time.monotonic()
                    for packet in self.reader.feed(data):
                        if packet['kind'] == 'shutdown':
                            self.done = True
                            return
                        if packet['kind'] == 'ack':
                            if packet['accepted']:
                                self.accepted += 1
                            else:
                                self.rejected += 1
                                self.get_logger().warning('Host rejected command: '+packet['error'])
                        elif packet['kind'] == 'api_ack':
                            self.api_acks[packet['request_id']] = packet
                            if packet['accepted']:
                                self.api_accepted += 1
                            else:
                                self.api_rejected += 1
                        elif packet['kind'] == 'state':
                            self.publish_state(packet)
                if time.monotonic()-self.last_receive > 2.:
                    raise ConnectionError('Host simulation stopped responding')
            except (OSError, ValueError, KeyError) as exc:
                self.get_logger().error(str(exc))
                self.done = True

        def publish_state(self, packet):
            if packet['session'] != args.session or packet['robot'] != args.robot:
                raise ValueError('Simulation session/robot mismatch')
            self.last = packet
            ns = round(packet['time_s']*1e9)
            stamp = Time(sec=ns//1000000000, nanosec=ns%1000000000)
            self.clock_pub.publish(Clock(clock=stamp))
            state = FullState()
            state.header.stamp, state.header.frame_id = stamp, packet['frame_id']
            p, q = packet['position_world_m'], packet['quaternion_xyzw']
            state.pose.position.x, state.pose.position.y, state.pose.position.z = p
            state.pose.orientation.x, state.pose.orientation.y, state.pose.orientation.z, state.pose.orientation.w = q
            state.twist.linear.x, state.twist.linear.y, state.twist.linear.z = packet['velocity_world_mps']
            state.twist.angular.x, state.twist.angular.y, state.twist.angular.z = packet['omega_body_radps']
            state.acc.x, state.acc.y, state.acc.z = packet['acceleration_world_mps2']
            self.state_pub.publish(state)
            self.pose_pub.publish(PoseStamped(header=state.header, pose=state.pose))
            odom = Odometry(header=state.header, child_frame_id=args.robot)
            odom.pose.pose = state.pose
            odom.twist.twist.angular = state.twist.angular
            odom.twist.twist.linear.x, odom.twist.twist.linear.y, odom.twist.twist.linear.z = body_velocity(packet['velocity_world_mps'], q)
            self.odom_pub.publish(odom)
            transform = TransformStamped(header=state.header, child_frame_id=args.robot)
            transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z = p
            transform.transform.rotation = state.pose.orientation
            self.tf.sendTransform(transform)
            cable = PoseArray(header=state.header)
            for xyz in packet['cable_positions_world_m']:
                pose = Pose()
                pose.position.x, pose.position.y, pose.position.z = xyz
                pose.orientation.w = 1.
                cable.poses.append(pose)
            self.cable_pub.publish(cable)
            status = {key:packet[key] for key in ('session','robot','frame_id','time_s','running','error','step_count','generation','command_delay_s','controller_period_s','output_period_s','commands_received','commands_scheduled')}
            status.update(schema=SCHEMA, source='Simulation flight', domain=int(os.environ.get('ROS_DOMAIN_ID', '0')),
                          commands_accepted=self.accepted, commands_rejected=self.rejected,
                          command_interface='crazyflie_airborne_api', command_error=self.command_error,
                          api_accepted=self.api_accepted, api_rejected=self.api_rejected,
                          initial_position=config['initial_position'], yaml_initial_position=config['yaml_initial_position'],
                          airborne_landing_height_m=config['airborne_landing_height_m'])
            status['acceleration_only'] = packet.get('acceleration_only', False)
            status['controller_interface'] = ('acceleration_direct_v1' if status['acceleration_only'] else 'pva')
            if self.get_parameter('aerowhip.acceleration_only').value != status['acceleration_only']:
                self.syncing_parameters = True
                try:
                    self.set_parameters([Parameter('aerowhip.acceleration_only', value=status['acceleration_only'])])
                finally:
                    self.syncing_parameters = False
            status['reference_position_world_m'] = (None if status['acceleration_only'] else packet['requested_command'][:3])
            for key in ('armed','emergency_latched','flight_mode','api_received','api_scheduled','high_level_samples','group_mask','late_stream_discarded','high_level_waiting_for_notify'):
                if key in packet:
                    status[key] = packet[key]
            self.status_pub.publish(String(data=json.dumps(status, allow_nan=False)))
            drone_status = Status(header=state.header)
            drone_status.supervisor_info = Status.SUPERVISOR_INFO_CAN_BE_ARMED
            if packet.get('armed'):
                drone_status.supervisor_info |= Status.SUPERVISOR_INFO_IS_ARMED | Status.SUPERVISOR_INFO_CAN_FLY
            if packet.get('emergency_latched'):
                drone_status.supervisor_info = Status.SUPERVISOR_INFO_IS_LOCKED
            self.drone_status_pub.publish(drone_status)

    bridge = None
    try:
        bridge = Bridge()
        while rclpy.ok() and not bridge.done:
            rclpy.spin_once(bridge, timeout_sec=.05)
    except KeyboardInterrupt:
        pass
    finally:
        if bridge is not None:
            bridge.peer.close()
            bridge.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
