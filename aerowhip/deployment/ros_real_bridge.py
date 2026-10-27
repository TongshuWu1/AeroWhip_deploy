"""Read-only OptiTrack pose relay, executed in the existing ROS container."""
import argparse
import json
import math
import socket
import struct
import time


SCHEMA = 'aerowhip_optitrack_display_v1'
MAX_PACKET_BYTES = 1048576


def cloud_xyz(message, maximum=2048):
    """Decode organized/padded XYZ32 clouds without relying on sensor_msgs_py.

    Preserve input ordering; a nonfinite coordinate is null in JSON evidence.
    Optional NatNet fields are decoded separately by cloud_marker_metadata.
    """
    if message.width*message.height > maximum:
        raise ValueError(f'Point cloud exceeds {maximum} points; no assignments made')
    fields = {field.name: field for field in message.fields}
    if any(name not in fields or fields[name].datatype != 7 or fields[name].count != 1
           for name in ('x', 'y', 'z')):
        raise ValueError('Point cloud requires scalar FLOAT32 x/y/z fields')
    offsets = [fields[name].offset for name in ('x', 'y', 'z')]
    if (any(offset < 0 or offset+4 > message.point_step for offset in offsets)
            or message.row_step < message.width*message.point_step
            or len(message.data) < message.row_step*message.height):
        raise ValueError('Invalid point-cloud layout')
    fmt = '>f' if message.is_bigendian else '<f'
    return [[value if math.isfinite(value) else None for value in
             (struct.unpack_from(fmt, message.data, row*message.row_step+col*message.point_step+offset)[0]
              for offset in offsets)]
            for row in range(message.height) for col in range(message.width)]


def cloud_marker_metadata(message):
    """Decode the patched capture node's aligned NatNet fields, if present."""
    fields = {field.name: field for field in message.fields}
    layout = (('natnet_id', 6, 'I', 4), ('natnet_flags', 4, 'H', 2),
              ('natnet_source', 2, 'B', 1))
    if not any(name in fields for name, *_ in layout):
        return None
    if (message.width*message.height > 2048
            or message.row_step < message.width*message.point_step
            or len(message.data) < message.row_step*message.height):
        raise ValueError('Invalid marker-metadata cloud layout/capacity')
    result = {}
    for name, datatype, fmt, size in layout:
        field = fields.get(name)
        if (field is None or field.datatype != datatype or field.count != 1
                or field.offset < 0 or field.offset+size > message.point_step):
            raise ValueError('Incomplete or invalid NatNet marker metadata')
        result[name] = [struct.unpack_from(('>' if message.is_bigendian else '<')+fmt,
            message.data, row*message.row_step+col*message.point_step+field.offset)[0]
            for row in range(message.height) for col in range(message.width)]
    return result


def cable_candidate_mask(metadata, count):
    """Observed, unassociated passive markers only; never infer cable ID here.

    Deprecated OtherMarkers can duplicate asset markers and have no membership,
    so remain recorded but ineligible. Modern PointCloud IDs can use all 32 bits;
    only legacy IDs encode asset membership in their high word.
    """
    if metadata is None:
        raise ValueError('Marker membership unavailable; rebuild the OptiTrack node with the marker-metadata patch')
    if not isinstance(metadata, dict):
        raise ValueError('Invalid NatNet marker metadata')
    for name, maximum in (('natnet_id', 0xffffffff), ('natnet_flags', 0xffff), ('natnet_source', 3)):
        values = metadata.get(name)
        if (not isinstance(values, list) or len(values) != count
                or any(type(v) is not int or not 0 <= v <= maximum for v in values)):
            raise ValueError('Invalid or unaligned NatNet marker metadata: '+name)
    return [
        (source == 3 and bool(flags & 0x10) and not flags & 0x2d)
        or (source == 2 and marker_id >> 16 == 0 and not flags & 0x05)
        for marker_id, flags, source in zip(metadata['natnet_id'],
            metadata['natnet_flags'], metadata['natnet_source'])]


def encode(packet):
    return (json.dumps(dict(packet, schema=SCHEMA), allow_nan=False,
                       separators=(',', ':'))+'\n').encode()


class PacketReader:
    def __init__(self):
        self.buffer = b''

    def feed(self, data):
        self.buffer += data
        packets = []
        while b'\n' in self.buffer:
            line, self.buffer = self.buffer.split(b'\n', 1)
            if len(line) > MAX_PACKET_BYTES:
                raise ValueError('Real-view packet exceeds size limit')
            packet = json.loads(line)
            if not isinstance(packet, dict) or packet.get('schema') != SCHEMA:
                raise ValueError('Incompatible real-view schema')
            packets.append(packet)
        if len(self.buffer) > MAX_PACKET_BYTES:
            raise ValueError('Real-view packet exceeds size limit')
        return packets


def pose_values(pose):
    p, q = pose.position, pose.orientation
    position = [float(p.x), float(p.y), float(p.z)]
    quaternion = [float(q.x), float(q.y), float(q.z), float(q.w)]
    if not all(math.isfinite(v) for v in position+quaternion):
        raise ValueError('Capture pose contains unavailable/nonfinite values')
    norm = math.sqrt(sum(v*v for v in quaternion))
    if abs(norm-1.) > .001:
        raise ValueError('Capture orientation is not a unit quaternion')
    return position, [v/norm for v in quaternion]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--session', required=True)
    parser.add_argument('--robots', required=True, help='JSON robot-name list')
    parser.add_argument('--frame', required=True)
    parser.add_argument('--capture-frame', required=True)
    parser.add_argument('--capture-node', default='/motion_capture_tracking')
    parser.add_argument('--poses-topic', default='/poses')
    parser.add_argument('--point-cloud-topic', help='Optional raw candidate-cloud relay')
    args = parser.parse_args()
    import rclpy
    from rclpy.node import Node
    from rclpy.parameter import Parameter
    from rclpy.parameter_client import AsyncParameterClient
    from rclpy.qos import qos_profile_sensor_data
    from rclpy.time import Time
    from motion_capture_tracking_interfaces.msg import NamedPoseArray
    from crazyflie_interfaces.msg import FullState, LogDataGeneric, Status
    from sensor_msgs.msg import PointCloud2
    from geometry_msgs.msg import Pose
    from tf2_ros import Buffer, TransformListener
    from tf2_geometry_msgs import do_transform_pose

    class CaptureRelay(Node):
        def __init__(self, peer):
            super().__init__('aerowhip_optitrack_view_'+args.session[:8],
                enable_rosout=False, start_parameter_services=False,
                parameter_overrides=[Parameter('use_sim_time', value=False)])
            self.peer = peer
            self.names = tuple(json.loads(args.robots))
            self.poses, self.issues = {}, {}
            self.pose_samples = []
            self.pose_samples_complete = True
            self.pending_clouds, self.pending_cloud_poses = {}, {}
            self.seen_names = []
            self.sequence = self.generation = 0
            self.flight_publishers = {name: set() for name in self.names}
            self.flight_epochs = {name: 0 for name in self.names}
            self.last_flight_check = -math.inf
            self.epoch_ns = self.previous_now_ns = self.get_clock().now().nanoseconds
            self.epoch = self.previous_now = self.epoch_ns/1e9
            self.capture_type = None
            self.publisher_gid = None
            self.last_capture_gid = None
            self.verification = None
            self.verification_started = 0.
            self.last_check = -math.inf
            self.buffer = Buffer(node=self)
            self.listener = TransformListener(self.buffer, self,
                                             qos=qos_profile_sensor_data)
            self.parameters = AsyncParameterClient(self, args.capture_node)
            self.subscription = self.create_subscription(NamedPoseArray,
                args.poses_topic, self.capture, qos_profile_sensor_data)
            # Battery and link state from each drone's status block (display only).
            self.status = {}
            self.status_subscriptions = [self.create_subscription(Status, f'/{name}/status',
                (lambda message, name=name: self.record_status(name, message)), qos_profile_sensor_data)
                for name in self.names]
            # Motor PWM and battery (motor_telemetry log block) and the acceleration command the host
            # sends (cmd_full_state), per drone, display only.
            self.motors, self.commands = {}, {}
            self.motor_subscriptions = [self.create_subscription(LogDataGeneric, f'/{name}/motor_telemetry',
                (lambda message, name=name: self.record_motors(name, message)), qos_profile_sensor_data)
                for name in self.names]
            self.command_subscriptions = [self.create_subscription(FullState, f'/{name}/cmd_full_state',
                (lambda message, name=name: self.record_command(name, message)), 10)
                for name in self.names]
            if args.point_cloud_topic:
                self.cloud_subscription = self.create_subscription(PointCloud2,
                    args.point_cloud_topic, self.capture_cloud, qos_profile_sensor_data)
            self.timer = self.create_timer(1/60, self.publish)
            self.send(dict(kind='ready', frame_id=args.frame, capture_frame=args.capture_frame,
                           capture_node=args.capture_node, poses_topic=args.poses_topic))

        def now_s(self):
            return self.get_clock().now().nanoseconds/1e9

        def record_motors(self, name, message):
            values = [float(v) for v in message.values]
            self.motors[name] = dict(pwm=values[:4], battery_v=values[4] if len(values) > 4 else None, time_s=self.now_s())

        def record_command(self, name, message):
            a, v, p = message.acc, message.twist.linear, message.pose.position
            self.commands[name] = dict(acceleration_mps2=[a.x, a.y, a.z], velocity_mps=[v.x, v.y, v.z],
                                       position_m=[p.x, p.y, p.z], time_s=self.now_s())

        def record_status(self, name, message):
            self.status[name] = dict(battery_v=float(message.battery_voltage), pm_state=int(message.pm_state),
                                     supervisor_info=int(message.supervisor_info), rssi=int(message.rssi),
                                     time_s=self.now_s())

        def send(self, packet):
            self.peer.sendall(encode(dict(packet, session=args.session)))

        def verify_capture(self):
            now = time.monotonic()
            if self.verification is not None:
                if self.verification.done():
                    try:
                        values = self.verification.result().values
                        capture_type, frame = values[0].string_value, values[1].string_value
                        if capture_type not in ('optitrack', 'optitrack_closed_source'):
                            raise ValueError(f'{args.capture_node} is not an OptiTrack source: {capture_type!r}')
                        if frame.lstrip('/') != args.capture_frame:
                            raise ValueError(f'Capture frame {frame!r} differs from loaded YAML {args.capture_frame!r}')
                        self.capture_type = capture_type
                        self.issues.pop('source', None)
                    except Exception as exc:
                        self.capture_type = None
                        self.issues['source'] = str(exc)
                    self.verification = None
                elif now-self.verification_started > 2.:
                    self.verification.cancel()
                    self.verification = None
                    self.capture_type = None
            if now-self.last_check < .5:
                return
            self.last_check = now
            publishers = self.get_publishers_info_by_topic(args.poses_topic)
            expected = args.capture_node.rstrip('/')
            owners = [(p.node_namespace.rstrip('/')+'/'+p.node_name) for p in publishers]
            if owners and (len(owners) != 1 or owners[0] != expected):
                self.capture_type = None
                self.publisher_gid = None
                if self.verification:
                    self.verification.cancel()
                    self.verification = None
                self.issues['source'] = f'Expected one OptiTrack publisher {expected}; found {owners}'
                return
            if not owners or not self.parameters.services_are_ready():
                self.capture_type = None
                self.publisher_gid = None
                if self.verification:
                    self.verification.cancel()
                    self.verification = None
                self.issues.pop('source', None)
                return
            gid = bytes(publishers[0].endpoint_gid)
            if gid != self.publisher_gid:
                if self.last_capture_gid is not None and gid != self.last_capture_gid:
                    self.poses.clear()
                    self.pending_clouds.clear()
                    self.pending_cloud_poses.clear()
                    self.pose_samples.clear()
                    self.pose_samples_complete = True
                    self.issues.pop('sample_stream', None)
                    self.generation += 1
                self.capture_type = None
                if self.verification:
                    self.verification.cancel()
                    self.verification = None
            self.last_capture_gid = gid
            self.publisher_gid = gid
            if self.verification is None:
                self.verification = self.parameters.get_parameters(['type', 'topics.frame_id'])
                self.verification_started = now

        def capture(self, message):
            if self.capture_type is None:
                return
            # Jazzy's callback metadata omits publisher_gid; require the sole
            # currently discovered endpoint to match the verified capture node.
            publishers = self.get_publishers_info_by_topic(args.poses_topic)
            if len(publishers) != 1 or bytes(publishers[0].endpoint_gid) != self.publisher_gid:
                return
            frame = message.header.frame_id.lstrip('/')
            stamp_ns = message.header.stamp.sec*10**9+message.header.stamp.nanosec
            stamp = stamp_ns/1e9
            names = [p.name for p in message.poses]
            self.seen_names = sorted(set(names))
            try:
                if frame != args.capture_frame or stamp_ns <= 0 or stamp_ns > self.get_clock().now().nanoseconds:
                    raise ValueError('Capture frame/timestamp differs from the configured causal ROS timeline')
                if len(names) != len(set(names)):
                    raise ValueError('Capture contains duplicate rigid-body names')
                transform = (None if frame == args.frame else self.buffer.lookup_transform(
                    args.frame, frame, Time.from_msg(message.header.stamp)))
                self.issues.pop('frame', None)
            except Exception as exc:
                self.issues['frame'] = str(exc)
                return
            matched = []
            for named in message.poses:
                if named.name not in self.names:
                    continue
                try:
                    pose_values(named.pose)  # Do not turn missing orientation into identity.
                    pose = named.pose if transform is None else do_transform_pose(named.pose, transform)
                    position, quaternion = pose_values(pose)
                    previous = self.poses.get(named.name)
                    if previous and stamp < previous['stamp_s']:
                        raise ValueError('Capture timestamp regressed')
                    self.poses[named.name] = dict(name=named.name, stamp_s=stamp,
                        stamp_ns=stamp_ns, time_s=(stamp_ns-self.epoch_ns)/1e9, position_world_m=position,
                        quaternion_xyzw=quaternion)
                    matched.append(self.poses[named.name])
                    if len(self.pose_samples) >= 512:
                        self.pose_samples_complete = False
                        self.issues['sample_stream'] = 'Capture sample buffer overflowed; cable simulation must stop'
                    else:
                        self.pose_samples.append(self.poses[named.name])
                    self.issues.pop(named.name, None)
                except Exception as exc:
                    self.issues[named.name] = str(exc)
            if args.point_cloud_topic:
                self.pending_cloud_poses[stamp_ns] = matched
                self.flush_clouds(stamp_ns)

        def capture_cloud(self, message):
            if self.capture_type is None:
                return
            pose_publishers = self.get_publishers_info_by_topic(args.poses_topic)
            if len(pose_publishers) != 1 or bytes(pose_publishers[0].endpoint_gid) != self.publisher_gid:
                return
            publishers = self.get_publishers_info_by_topic(args.point_cloud_topic)
            owners = [p.node_namespace.rstrip('/')+'/'+p.node_name for p in publishers]
            if owners != [args.capture_node.rstrip('/')]:
                self.issues['point_cloud'] = 'Expected one cloud publisher from the verified capture node'
                return
            try:
                frame = message.header.frame_id.lstrip('/')
                stamp_ns = message.header.stamp.sec*10**9+message.header.stamp.nanosec
                if frame != args.capture_frame or not 0 < stamp_ns <= self.get_clock().now().nanoseconds:
                    raise ValueError('Point-cloud frame/timestamp mismatch')
                raw = cloud_xyz(message)
                metadata = cloud_marker_metadata(message)
                transformed = raw
                if frame != args.frame:
                    transform = self.buffer.lookup_transform(args.frame, frame, Time.from_msg(message.header.stamp))
                    transformed = []
                    for xyz in raw:
                        if None in xyz:
                            transformed.append([None, None, None])
                            continue
                        pose = Pose()
                        pose.orientation.w = 1.
                        pose.position.x, pose.position.y, pose.position.z = xyz
                        point = do_transform_pose(pose, transform).position
                        transformed.append([point.x, point.y, point.z])
                self.pending_clouds[stamp_ns] = dict(raw_points_capture_m=raw,
                    points_world_m=transformed, capture_frame=frame,
                    point_metadata=metadata,
                    source='Mixed NatNet OtherMarkers + LabeledMarkers; aligned metadata when supported')
                self.issues.pop('point_cloud', None)
                self.flush_clouds(stamp_ns)
            except Exception as exc:
                self.issues['point_cloud'] = str(exc)

        def flush_clouds(self, stamp_ns):
            # Both upstream messages use the SAME capture-node receipt timestamp.
            # Never substitute a nearby/current drone pose for a missing match.
            for stamp in sorted(set(self.pending_clouds) & set(self.pending_cloud_poses)):
                self.send(dict(kind='markers', frame_id=args.frame, generation=self.generation,
                    stamp_ns=stamp, time_s=(stamp-self.epoch_ns)/1e9,
                    robots=self.pending_cloud_poses.pop(stamp),
                    **self.pending_clouds.pop(stamp)))
            for stamp in sorted(self.pending_clouds):
                if stamp_ns-stamp > 100_000_000 or len(self.pending_clouds) > 32:
                    self.send(dict(kind='markers', frame_id=args.frame, generation=self.generation,
                        stamp_ns=stamp, time_s=(stamp-self.epoch_ns)/1e9, robots=[],
                        pairing_issue='No matching pose frame; no reconstruction',
                        **self.pending_clouds.pop(stamp)))
            for stamp in list(self.pending_cloud_poses):
                if stamp_ns-stamp > 100_000_000 or len(self.pending_cloud_poses) > 32:
                    self.pending_cloud_poses.pop(stamp)

        def publish(self):
            if time.monotonic()-self.last_flight_check >= .1:
                self.last_flight_check = time.monotonic()
                for name in self.names:
                    publishers = {bytes(p.endpoint_gid) for topic in
                        (f'/{name}/cmd_full_state', '/all/cmd_full_state')
                        for p in self.get_publishers_info_by_topic(topic)}
                    if publishers-self.flight_publishers[name]:
                        self.flight_epochs[name] += 1
                    self.flight_publishers[name] = publishers
            now_ns = self.get_clock().now().nanoseconds
            now = now_ns/1e9
            if now_ns < self.previous_now_ns:
                self.poses.clear()
                self.pending_clouds.clear()
                self.pending_cloud_poses.clear()
                self.pose_samples.clear()
                self.pose_samples_complete = True
                self.issues.pop('sample_stream', None)
                self.epoch_ns = now_ns
                self.epoch = now
                self.generation += 1
            self.previous_now_ns = now_ns
            self.previous_now = now
            self.verify_capture()
            self.sequence += 1
            self.send(dict(kind='state', sequence=self.sequence, generation=self.generation,
                frame_id=args.frame, time_s=(now_ns-self.epoch_ns)/1e9, ros_time_s=now,
                capture_type=self.capture_type, capture_frame=args.capture_frame,
                poses_topic=args.poses_topic, robots=list(self.poses.values()),
                pose_samples=self.pose_samples, pose_samples_complete=self.pose_samples_complete,
                flight_epochs=self.flight_epochs,
                seen_names=self.seen_names, missing_robots=[n for n in self.names if n not in self.poses],
                status={n: dict(v, age_s=now-v['time_s']) for n, v in self.status.items()},
                motors={n: dict(v, age_s=now-v['time_s']) for n, v in self.motors.items()},
                commands={n: dict(v, age_s=now-v['time_s']) for n, v in self.commands.items()},
                issues=self.issues))
            self.pose_samples.clear()

    rclpy.init(args=[])
    peer = socket.create_connection(('127.0.0.1', args.port), timeout=5.)
    peer.settimeout(2.)
    node = None
    try:
        node = CaptureRelay(peer)
        rclpy.spin(node)
    except (BrokenPipeError, ConnectionResetError, TimeoutError, KeyboardInterrupt):
        pass  # Viewer disconnect affects this reader only, never the flight server.
    finally:
        if node:
            node.destroy_node()
        peer.close()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
