"""Run the real Crazyswarm/OptiTrack stack with the shared read-only flight viewer."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # Repository root: runs without installing.
from aerowhip.paths import ROOT
import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
import math
import re
import signal
import subprocess
import time
from threading import Event
import uuid

from aerowhip.deployment.configuration import parse_yaml_in_container
from aerowhip.deployment.cable_tracking import TrackingSettings
from aerowhip.deployment.custom_simulation import Evidence, launch_container_script
from aerowhip.deployment.real_flight import RealFlightFeed, RealFlightTransport, existing_real_nodes, select_real_robots, stop_real_stack
from aerowhip.deployment.stack_lifecycle import capture_real_stack, close_captured_stack
from aerowhip.paths import project_path
from aerowhip.physics.configuration import load_configuration
from aerowhip.physics.configuration import DEFAULT_MODEL
from aerowhip.physics.live_cable import LiveCable
from aerowhip.physics.model import Model


CRAZYSWARM = ROOT.parent/'Crazyflie/crazyswarm2_jazzy_whipping'


def supervise_viewer(args, argv):
    """Keep native Qt/VTK crashes outside the process owning shutdown."""
    session = uuid.uuid4().hex
    stop = Event()
    handlers = {sig: signal.signal(sig, lambda *_: stop.set())
                for sig in (signal.SIGINT, signal.SIGTERM)}
    child = attached_stack = None
    try:
        if args.attach:
            attached_stack = capture_real_stack(args.container, args.domain)
        child = subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()),
            *argv, '--viewer-worker', '--session', session], start_new_session=True)
        while child.poll() is None and not stop.wait(.1):
            pass
        if child.poll() is None:
            child.send_signal(signal.SIGINT)
            try:
                child.wait(30.)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(5.)
        elif child.returncode < 0 and not stop.is_set() and not args.attach and not getattr(args, 'capture_only', False):
            print('Viewer process crashed. The real ROS stack remains owned by this terminal. '
                  'Use --attach in another terminal to reopen the viewer. '
                  'After flight has ended, Ctrl+C here stops this stack.', flush=True)
            while not stop.wait(.1):
                pass
        return child.returncode if child.returncode >= 0 else 128-child.returncode
    finally:
        # Even a native crash in the worker cannot skip ownership-checked cleanup.
        if args.attach and stop.is_set():
            close_captured_stack(args.container, args.domain, attached_stack)
        if child is not None and not args.attach:
            for signum, timeout in ((signal.SIGINT, 10.), (signal.SIGTERM, 5.), (signal.SIGKILL, 3.)):
                if not stop_real_stack(args.container, session, int(signum)):
                    break
                deadline = time.monotonic()+timeout
                while time.monotonic() < deadline:
                    if not stop_real_stack(args.container, session, 0):
                        break
                    time.sleep(.2)
                else:
                    continue
                break
        for sig, handler in handlers.items():
            signal.signal(sig, handler)


HEARTBEATS = CRAZYSWARM/'logs'/'aerowhip_heartbeat'     # bind-mounted as /workspace/logs/aerowhip_heartbeat


def start_heartbeat(session):
    """Touch a per-session file every 0.5 s for as long as this viewer process lives; the stack it
    launches in the container stops itself when the file goes stale (see real_stack.watch_heartbeat).
    Returns the file's path inside the container."""
    import atexit
    import threading
    import time
    HEARTBEATS.mkdir(parents=True, exist_ok=True)
    path = HEARTBEATS/session
    path.touch()

    def beat():
        while True:
            path.touch()
            time.sleep(.5)
    threading.Thread(target=beat, daemon=True).start()
    atexit.register(lambda: path.unlink(missing_ok=True))
    return f'/workspace/logs/aerowhip_heartbeat/{session}'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--container', default='crazyswarm2_jazzy_whipping_dev')
    parser.add_argument('--crazyflies-yaml', type=Path, default=CRAZYSWARM/'config/crazyflies.yaml')
    parser.add_argument('--motion-capture-yaml', type=Path, default=CRAZYSWARM/'config/motion_capture.yaml')
    parser.add_argument('--attach', action='store_true', help='Use an existing real stack; Ctrl+C also stops its local ROS launch')
    parser.add_argument('--capture-only', action='store_true', help='Hand-movement test: start only OptiTrack, enable cable tracking, close capture on viewer exit')
    parser.add_argument('--record', action='store_true', help='Start a recording immediately; otherwise use Record in the viewer')
    parser.add_argument('--domain', type=int, help='ROS domain: default 43 for --capture-only, 42 otherwise; non-42 domains require --attach or --capture-only')
    parser.add_argument('--robot', action='append', help='Display selected enabled robot(s); default is all enabled')
    parser.add_argument('--frame', help='Display frame; default is the OptiTrack YAML frame; other frames require TF')
    parser.add_argument('--poses-topic', default='/poses')
    parser.add_argument('--track-cable', action='store_true', help='Experimental cable association from the raw cloud; select one robot and initialize hanging')
    parser.add_argument('--point-cloud-topic', default='/pointCloud')
    parser.add_argument('--cable-tracking-config', type=Path, help='JSON tracking settings; default marker arc distances come from the selected model')
    parser.add_argument('--capture-node', default='/motion_capture_tracking')
    parser.add_argument('--config', type=Path, default=ROOT/'configs/scenes/default.toml', help='Existing viewer/world geometry settings')
    parser.add_argument('--model', type=Path, default=DEFAULT_MODEL, help='Existing physical model for the measured-motion cable simulation')
    parser.add_argument('--no-cable', action='store_true', help='Drone without cable: show only the drone, no simulated cable')
    parser.add_argument('--port', type=int, default=0, help='Loopback viewer port; zero chooses an unused port')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--duration', type=float, help='Headless wall-time limit in seconds')
    parser.add_argument('--viewer-worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--session', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.domain is None:
        args.domain = 43 if args.capture_only else 42
    if args.capture_only and args.attach:
        parser.error('Use --capture-only for a new capture node, or --attach --track-cable for an existing stack')
    if args.capture_only:
        args.track_cable = True
    if not args.headless and not args.viewer_worker:
        return supervise_viewer(args, sys.argv[1:])
    if not 0 <= args.domain <= 232 or (not args.attach and not args.capture_only and args.domain != 42):
        parser.error('Real flight stack launches use domain 42; --attach or --capture-only can select a domain in 0..232')
    if not 0 <= args.port <= 65535:
        parser.error('--port must be in 0..65535')
    if args.duration is not None and (not args.headless or not math.isfinite(args.duration) or args.duration <= 0):
        parser.error('--duration requires --headless and a positive finite duration')
    if any(not re.fullmatch(r'/?[A-Za-z_][A-Za-z0-9_]*(?:/[A-Za-z_][A-Za-z0-9_]*)*', value)
           for value in (args.poses_topic, args.capture_node, args.point_cloud_topic)):
        parser.error('Use valid ROS names for --poses-topic and --capture-node')
    args.poses_topic = '/'+args.poses_topic.lstrip('/')
    args.capture_node = '/'+args.capture_node.lstrip('/')
    args.point_cloud_topic = '/'+args.point_cloud_topic.lstrip('/')
    if args.no_cable and args.track_cable:
        parser.error('--track-cable needs the cable; drop --no-cable')
    if args.cable_tracking_config and not args.track_cable:
        parser.error('--cable-tracking-config requires --track-cable')
    if not args.attach and (args.poses_topic != '/poses' or args.capture_node != '/motion_capture_tracking'):
        parser.error('Custom capture topic/node names require --attach to the corresponding existing stack')
    running = subprocess.run(['docker', 'inspect', '--format', '{{.State.Running}}', args.container],
                             capture_output=True, text=True)
    if running.returncode or running.stdout.strip() != 'true':
        parser.error('ROS container is not running; start ./docker/compose in Crazyswarm first')
    try:
        paths = [args.crazyflies_yaml.resolve(), args.motion_capture_yaml.resolve()]
        original_yaml = [path.read_text(encoding='utf-8') for path in paths]
        base, capture = parse_yaml_in_container(args.container, original_yaml)
        robots = select_real_robots(base, args.robot)
        if args.track_cable and len(robots) != 1:
            raise ValueError('--track-cable requires one selected robot; use --robot NAME')
        capture = capture['/motion_capture_tracking']['ros__parameters']
        if capture['type'] not in ('optitrack', 'optitrack_closed_source'):
            raise ValueError('Custom real requires the actual OptiTrack capture configuration')
        capture_frame = capture.get('topics', {}).get('frame_id', 'world').lstrip('/')
        frame = (args.frame or capture_frame).lstrip('/')
        if not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_/]*', frame) or not capture_frame:
            raise ValueError('Capture/display frame names must be nonempty ROS frame names')
        configuration = load_configuration(project_path(args.config))
        configuration = replace(configuration,
            viewer=replace(configuration.viewer, visual_rotor_rpm=0.))
    except (ValueError, KeyError, OSError, subprocess.CalledProcessError) as exc:
        parser.error(str(exc))
    session = args.session or uuid.uuid4().hex
    if not re.fullmatch(r'[0-9a-f]{32}', session):
        parser.error('Invalid internal session identity')
    output = project_path(args.output) if args.output else ROOT/'output/custom_real'/f'{datetime.now(timezone.utc):%Y%m%d-%H%M%S}-{session[:8]}'
    evidence = Evidence(output)
    feed = transport = relay = stack = capture_log = stack_log = lease = viewer = attached_stack = None
    stop = Event()

    def request_stop(signum, context):
        stop.set()

    previous_handlers = {sig: signal.signal(sig, request_stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        if args.attach:
            attached_stack = capture_real_stack(args.container, args.domain)
            if attached_stack is None:
                print('No matching local real ROS launch found. Ctrl+C can stop this viewer only.', flush=True)
        if not args.attach:
            import fcntl
            lease = Path(f'/tmp/aerowhip-real-launch-domain-{args.domain}.lock').open('a')
            try:
                fcntl.flock(lease.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError('A custom real stack is already launched; use --attach') from None
            nodes = existing_real_nodes(args.container, args.domain)
            for _ in range(20):      # a stack left by a closed viewer stops itself within ~3 s of its heartbeat ending
                if not nodes:
                    break
                print(f'Waiting for a previous stack to stop: {nodes}', flush=True)
                time.sleep(.5)
                nodes = existing_real_nodes(args.container, args.domain)
            if nodes:
                raise RuntimeError(f'Real ROS nodes already exist: {nodes}. Use --attach; no existing process was stopped.')
        model_path = project_path(args.model)
        cable_model = None if args.no_cable else Model(model_path)
        tracking_settings = (TrackingSettings.load(
            project_path(args.cable_tracking_config) if args.cable_tracking_config else None,
            cable_model.config['cable']['marker_interval_lengths_m']) if args.track_cable else None)
        feed = RealFlightFeed(configuration, robots, frame, session, cable_model=cable_model,
            tracking_settings=tracking_settings, capture_only=args.capture_only)
        transport = RealFlightTransport(feed, evidence, port=args.port)
        metadata = dict(session=session, mode='capture-only' if args.capture_only else 'attached-viewer' if args.attach else 'real-stack-and-viewer',
            cable_tracking=None if tracking_settings is None else dict(settings=asdict(tracking_settings),
                topic=args.point_cloud_topic, initialization='Complete approximately vertical hanging cable',
                source='Mixed raw XYZ with aligned NatNet IDs/flags/source; observed unassociated passive candidates only',
                status='Experimental association overlay; heuristic confidence; no simulation correction'),
            shutdown_target=attached_stack,
            cable_simulation=None if cable_model is None else dict(model=str(model_path), configuration=cable_model.config,
                model_sources=cable_model.configuration_sources,
                initial_condition='Assumed hanging, stationary, untwisted at first measured pose',
                boundary='Causal attachment Hermite P/V/A and piecewise SO(3) attitude interpolation',
                maximum_gap_s=.25,
                numerical_refinement_factors=list(LiveCable.refinement_factors),
                numerical_refinement=('Multiply baseline interval substep count; retry endpoint, '
                    'position and velocity numerical rejections from unchanged history'),
                status='Simulated cable; no measurement correction or real-flight validation'),
            domain=args.domain, container=args.container, robots=robots, frame_id=frame,
            capture_frame=capture_frame, poses_topic=args.poses_topic, capture_node=args.capture_node,
            host_port=transport.port, yaml_sources=[dict(path=str(p), sha256=hashlib.sha256(raw.encode()).hexdigest())
                for p, raw in zip(paths, original_yaml)],
            viewer_scene=str(project_path(args.config)),
            data_source=f'OptiTrack {args.poses_topic} only; no estimator fallback',
            adapter_hashes={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (Path(__file__), ROOT/'aerowhip/deployment/real_flight.py', ROOT/'aerowhip/deployment/ros_real_bridge.py',
                          ROOT/'aerowhip/deployment/real_stack.py', ROOT/'aerowhip/deployment/configuration.py',
                          ROOT/'aerowhip/deployment/custom_simulation.py', ROOT/'aerowhip/deployment/stack_lifecycle.py',
                          ROOT/'aerowhip/physics/live_cable.py', ROOT/'aerowhip/deployment/cable_tracking.py',
                          ROOT/'aerowhip/viewer/cable_tracking_scene.py', ROOT/'aerowhip/viewer/display_state.py',
                          ROOT/'aerowhip/viewer/recording.py')},
            status='Display connection evidence; not a full-rate source recording, controller fit or real-flight validation')
        with (output/'metadata.json').open('x') as handle:
            json.dump(metadata, handle, indent=2, allow_nan=False)
            handle.write('\n')
        feed.configure_recording(output/'recordings', metadata)
        if args.record:
            print(f'Recording: {feed.start_recording()}', flush=True)
        if not args.attach:
            stack_log = (output/'real-stack.log').open('x')
            heartbeat = start_heartbeat(session)
            stack = launch_container_script(args.container, ROOT/'aerowhip/deployment/real_stack.py',
                ['--session', session, '--configuration', json.dumps(dict(zip(('crazyflies', 'motion_capture'), original_yaml))),
                 '--heartbeat', heartbeat]
                + (['--capture-only'] if args.capture_only else []),
                domain=args.domain, log=stack_log, start_new_session=True)
        capture_log = (output/'ros-capture.log').open('x')
        relay = launch_container_script(args.container, ROOT/'aerowhip/deployment/ros_real_bridge.py',
            ['--session', session, '--port', transport.port, '--robots', json.dumps([r['name'] for r in robots]),
             '--frame', frame, '--capture-frame', capture_frame,
             '--poses-topic', args.poses_topic, '--capture-node', args.capture_node]
            + (['--point-cloud-topic', args.point_cloud_topic] if args.track_cable else []),
            domain=args.domain, log=capture_log, start_new_session=True)
        if not transport.connected.wait(10.):
            raise RuntimeError(f'Capture relay did not connect; inspect {output/"ros-capture.log"}')
        if stop.is_set():
            return
        print(f'{feed.source_label} · {args.poses_topic} · frame {frame} · domain {args.domain}', flush=True)
        print(f'Robots: {", ".join(r["name"] for r in robots)}. Viewer is read-only.'+
              (' Move the tracked drone by hand.' if args.capture_only else ' Use the existing flight scripts.'), flush=True)
        print(f'Viewer connection evidence: {output}', flush=True)
        if args.capture_only:
            print('Capture-only hand test: no Crazyflie flight server was started. '
                  'Use Record / Stop & save; closing the viewer also stops this capture node.', flush=True)
            print('Capture-only display: measured drone and accepted blue cable estimates. '
                  'Raw candidates, simulated cable and simulated tip trail start hidden; enable them in Displays. '
                  'Hidden layers are still recorded.', flush=True)
        print('Cable simulation follows measured attachment motion; initial cable is assumed hanging/untwisted. '
              'The displayed floor has no contact response. Tracking gaps over 0.25 s freeze the cable.', flush=True)
        if tracking_settings is not None:
            print('Cable tracking: hold the cable fully visible and approximately vertical to initialize. '
                  'Gray = raw candidates; blue = estimated cable identities/observed links; green = simulation. '
                  'Missing/ambiguous nodes are not filled. Cloud and assignment records are in connection.jsonl.', flush=True)
            print('Candidate filter: rigid-body/asset markers, occluded/model-solved markers and '
                  'unclassified legacy duplicates are excluded; raw observations remain recorded. '
                  'Requires the patched OptiTrack node with marker metadata.', flush=True)
            print('Marker arc distances from attachment (m): '+', '.join(
                f'{value:g}' for value in tracking_settings.marker_distances_m), flush=True)
        if args.headless:
            import time
            started = time.monotonic()
            while not stop.wait(.05) and (args.duration is None or time.monotonic()-started < args.duration):
                if transport.error or relay.poll() is not None or (stack and stack.poll() is not None):
                    raise RuntimeError(transport.error or f'A ROS child exited; inspect logs in {output}')
        else:
            from aerowhip.viewer.display_state import WorldDisplay
            from aerowhip.viewer.pyvista_viewer import PyVistaViewer
            viewer = PyVistaViewer(configuration,
                world=replace(WorldDisplay.from_configuration(configuration.viewer), frame_id=frame))
            if args.capture_only:
                for robot in robots:
                    viewer.set_display_visible('raw_cloud', False, robot=robot['name'])
                    viewer.set_display_visible('cable', False, robot=robot['name'])
                viewer.set_overlay_visible('tip_trail', False)
            viewer.stop_requested = stop
            viewer.show_source(feed)
            feed.close()
            if stack and not args.capture_only and not stop.is_set() and stack.poll() is None:
                print('Viewer closed. Real ROS stack remains active, as with RViz. Ctrl+C here stops this launch.', flush=True)
                while stack.poll() is None and not stop.wait(.1):
                    pass
    except (RuntimeError, ValueError, OSError, subprocess.CalledProcessError) as exc:
        parser.exit(status=1, message=f'Custom real: {exc}\n')
    finally:
        if feed:
            feed.close()
        elif transport:
            transport.close()
        if relay:
            try:
                relay.wait(5.)
            except subprocess.TimeoutExpired:
                relay.terminate()
                relay.wait(5.)
        if stack:
            for signum, timeout in ((signal.SIGINT, 10.), (signal.SIGTERM, 5.), (signal.SIGKILL, 3.)):
                stop_real_stack(args.container, session, int(signum))
                try:
                    stack.wait(timeout)
                    break
                except subprocess.TimeoutExpired:
                    continue
        if args.attach and stop.is_set():
            close_captured_stack(args.container, args.domain, attached_stack)
        for handle in (capture_log, stack_log, lease):
            if handle:
                handle.close()
        evidence.close()
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)


if __name__ == '__main__':
    raise SystemExit(main())
