"""Simulation-only scripted PVA test through Crazyswarm's cmdFullState serializer.

Requires a running custom simulation. Bounded waits and a session marker prevent
this script from treating a real-flight server as a custom simulation.
"""
import argparse
import json
import math
import os
from pathlib import Path
import time


def main():
    import rclpy
    from crazyflie_interfaces.msg import FullState
    from crazyflie_py.crazyflie import Crazyflie
    from geometry_msgs.msg import PoseArray, PoseStamped
    from nav_msgs.msg import Odometry
    from rclpy.node import Node
    from rclpy.parameter import Parameter
    from std_msgs.msg import String
    from rosgraph_msgs.msg import Clock
    from tf2_msgs.msg import TFMessage

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--robot', default='cf_2')
    parser.add_argument('--rate', type=float, default=30.)
    parser.add_argument('--motion-duration', type=float, default=6.)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if int(os.environ.get('ROS_DOMAIN_ID','0')) == 42:
        parser.error('The simulation test cannot use real-flight ROS domain 42')
    if not math.isfinite(args.rate) or not 1 <= args.rate <= 100:
        parser.error('Command rate must be in 1..100 Hz')
    if not math.isfinite(args.motion_duration) or args.motion_duration <= 0:
        parser.error('Motion duration must be positive and finite')
    rclpy.init()
    node = Node('aerowhip_pva_test', parameter_overrides=[Parameter('use_sim_time', value=True)])
    status, states, poses, sent = {}, [], [], []
    clocks, odometry, cables, transforms = [], [], [], []
    status_wall = [0.]

    def receive_status(message):
        value = json.loads(message.data)
        if value.get('schema') == 'aerowhip_ros_stream_v1' and value.get('robot') == args.robot:
            status.clear(); status.update(value); status_wall[0] = time.monotonic()

    def receive_state(message):
        p, v, q, a, w = message.pose.position, message.twist.linear, message.pose.orientation, message.acc, message.twist.angular
        states.append(dict(time_s=message.header.stamp.sec+message.header.stamp.nanosec*1e-9,
            frame_id=message.header.frame_id, position_world_m=[p.x,p.y,p.z], velocity_world_mps=[v.x,v.y,v.z],
            quaternion_xyzw=[q.x,q.y,q.z,q.w], acceleration_world_mps2=[a.x,a.y,a.z], omega_body_radps=[w.x,w.y,w.z]))

    def receive_pose(message):
        p = message.pose.position
        poses.append(dict(time_s=message.header.stamp.sec+message.header.stamp.nanosec*1e-9,
                          frame_id=message.header.frame_id, position_world_m=[p.x,p.y,p.z]))

    def receive_odometry(message):
        p, q, v, w = message.pose.pose.position, message.pose.pose.orientation, message.twist.twist.linear, message.twist.twist.angular
        odometry.append(dict(time_s=message.header.stamp.sec+message.header.stamp.nanosec*1e-9,
            frame_id=message.header.frame_id, child_frame_id=message.child_frame_id,
            position_world_m=[p.x,p.y,p.z], quaternion_xyzw=[q.x,q.y,q.z,q.w],
            velocity_body_mps=[v.x,v.y,v.z], omega_body_radps=[w.x,w.y,w.z]))

    def receive_cable(message):
        cables.append(dict(time_s=message.header.stamp.sec+message.header.stamp.nanosec*1e-9,
            frame_id=message.header.frame_id, positions_world_m=[[p.position.x,p.position.y,p.position.z] for p in message.poses]))

    def receive_tf(message):
        for transform in message.transforms:
            if transform.child_frame_id != args.robot:
                continue
            p, q = transform.transform.translation, transform.transform.rotation
            transforms.append(dict(time_s=transform.header.stamp.sec+transform.header.stamp.nanosec*1e-9,
                frame_id=transform.header.frame_id, child_frame_id=transform.child_frame_id,
                position_world_m=[p.x,p.y,p.z], quaternion_xyzw=[q.x,q.y,q.z,q.w]))

    node.create_subscription(String, '/aerowhip/simulation/status', receive_status, 10)
    node.create_subscription(FullState, f'/{args.robot}/simulation_state', receive_state, 100)
    node.create_subscription(PoseStamped, f'/{args.robot}/pose', receive_pose, 100)
    node.create_subscription(Odometry, f'/{args.robot}/odom', receive_odometry, 100)
    node.create_subscription(PoseArray, f'/{args.robot}/simulation_cable', receive_cable, 100)
    node.create_subscription(TFMessage, '/tf', receive_tf, 100)
    node.create_subscription(Clock, '/clock', lambda m: clocks.append(m.clock.sec+m.clock.nanosec*1e-9), 100)

    class FullStateStream(Crazyflie):
        """Only the upstream streaming method; high-level services are not used."""
        def __init__(self):
            self.node = node
            self.cmdFullStatePublisher = node.create_publisher(FullState, f'/{args.robot}/cmd_full_state', 1)
            self.cmdFullStateMsg = FullState()
            self.cmdFullStateMsg.header.frame_id = '/world'

    client = FullStateStream()
    failure = None
    session = None
    try:
        deadline = time.monotonic()+10.
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=.02)
            if status.get('running') and states and poses and odometry and cables and transforms and clocks and node.get_clock().now().nanoseconds > 0 and client.cmdFullStatePublisher.get_subscription_count():
                break
        else:
            raise RuntimeError('No running custom simulation/clock/feedback within 10 s; launch --custom-sim first')
        session = status['session']
        generation = status['generation']
        received_before, rejected_before = status['commands_received'], status['commands_rejected']
        start = node.get_clock().now().nanoseconds*1e-9
        initial = list(states[-1]['position_world_m'])
        # 1 s initial hold, sin^4 outward/return excursion with zero P/V/A at
        # both endpoints, 2 s final hold. These are transport-test settings.
        total = 1.+args.motion_duration+2.
        next_time = start
        last_clock, last_advance = start, time.monotonic()
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=.001)
            now = node.get_clock().now().nanoseconds*1e-9
            if now < last_clock-1e-9:
                raise RuntimeError('Simulation clock reset during the test')
            if now > last_clock+1e-9:
                last_clock, last_advance = now, time.monotonic()
            if time.monotonic()-last_advance > 2. or time.monotonic()-status_wall[0] > 2.:
                raise RuntimeError('Simulation clock/feedback stalled')
            if status.get('session') != session or status.get('generation') != generation or not status.get('running') or status.get('error'):
                raise RuntimeError('Simulation session stopped, changed or reported an error')
            elapsed = now-start
            if elapsed >= total:
                break
            if now+1e-9 < next_time:
                continue
            missed = max(0, math.floor((now-next_time)*args.rate+1e-9))
            next_time += (missed+1)/args.rate
            u = elapsed-1.
            if 0. < u < args.motion_duration:
                k = math.pi/args.motion_duration
                s, c = math.sin(k*u), math.cos(k*u)
                b, db, ddb = s**4, 4*k*s**3*c, 4*k*k*(3*s*s*c*c-s**4)
            else:
                b = db = ddb = 0.
            amplitude = [.20, .10, .05]
            p = [x+scale*b for x,scale in zip(initial, amplitude)]
            v, a = [scale*db for scale in amplitude], [scale*ddb for scale in amplitude]
            client.cmdFullState(p, v, a, 0., [0.,0.,0.])
            sent.append(dict(stamp_s=now, elapsed_s=elapsed, command=p+v+a+[0.,0.,0.,0.], missed_slots=missed))
        # Drain acknowledgements and delayed final command on the simulation clock.
        end = time.monotonic()+.2
        while time.monotonic() < end:
            rclpy.spin_once(node, timeout_sec=.005)
        if status.get('commands_rejected',0) != rejected_before:
            raise RuntimeError('Adapter rejected a command during the test')
        if status.get('commands_received',0)-received_before != len(sent):
            raise RuntimeError('Accepted ROS command count does not match the sent count')
        if status.get('commands_scheduled',0) != status.get('commands_received',0):
            raise RuntimeError('Accepted commands remain unscheduled')
    except (Exception, KeyboardInterrupt) as exc:
        failure = str(exc) or type(exc).__name__
    finally:
        result = dict(session=session, robot=args.robot, domain=int(os.environ.get('ROS_DOMAIN_ID','0')),
                      failure=failure, status=status, sent=sent, states=states, poses=poses,
                      odometry=odometry, cables=cables, transforms=transforms, clocks=clocks,
                      command_rate_hz=args.rate, motion_duration_s=args.motion_duration,
                      interpretation='ROS connection/tracking evidence; not a real-flight or firmware-equivalence test')
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open('x') as file:
            json.dump(result, file, indent=2, allow_nan=False)
            file.write('\n')
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    print(json.dumps(dict(output=args.output, commands=len(sent), feedback_samples=len(states), failure=failure)), flush=True)
    if failure:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
