"""Launch/own the stock real ROS stack; no flight commands or parameter edits."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile


def capture_configuration(configuration):
    """Match stock Crazyswarm's capture-only parameter construction."""
    import yaml
    crazyflies = yaml.safe_load(configuration['crazyflies'])
    capture = yaml.safe_load(configuration['motion_capture'])
    params = capture['/motion_capture_tracking']['ros__parameters']
    params['rigid_bodies'] = {}
    version = crazyflies.get('fileversion', 1)
    for name, robot in crazyflies['robots'].items():
        tracking = crazyflies['robot_types'][robot['type']]['motion_capture']
        if robot['enabled'] and ((version == 1 and tracking.get('enabled')) or
                                (version >= 2 and tracking.get('tracking') == 'librigidbodytracker')):
            params['rigid_bodies'][name] = dict(initial_position=robot['initial_position'],
                marker=tracking['marker'], dynamics=tracking['dynamics'])
    # The launch_ros parameter flattener omits empty maps; ROS's direct YAML
    # parser instead rejects them. Vendor-only capture has no tracker bodies.
    if not params['rigid_bodies']:
        del params['rigid_bodies']
    return yaml.safe_dump(capture)


def stack_command(paths, capture_only=False):
    if capture_only:
        return ['ros2', 'run', 'motion_capture_tracking', 'motion_capture_tracking_node',
            '--ros-args', '-r', '__node:=motion_capture_tracking',
            '--params-file', str(paths['motion_capture'])]
    return ['ros2', 'launch', 'crazyflie', 'launch.py',
        'backend:=cflib', 'mocap:=True', 'teleop:=False', 'rviz:=False',
        f'crazyflies_yaml_file:={paths["crazyflies"]}',
        f'motion_capture_yaml_file:={paths["motion_capture"]}']


def wait_for_radios(crazyflies_yaml, timeout_s=90.):
    """Block until every enabled Crazyflie answers a radio scan (a drone that was just powered
    takes seconds to boot; launching before that makes the server lose the link at once and
    die). Returns the URIs found; raises after timeout_s with the ones still missing."""
    import time
    import yaml
    import cflib.crtp
    robots = yaml.safe_load(crazyflies_yaml)['robots']
    wanted = {name: robot['uri'].lower() for name, robot in robots.items() if robot.get('enabled')}
    if not wanted:
        return {}
    cflib.crtp.init_drivers()
    deadline, last = time.monotonic()+timeout_s, 0.
    missing = dict(wanted)
    while missing:
        for name, uri in list(missing.items()):
            address = int(uri.rsplit('/', 1)[-1], 16)
            if any(found.lower() == uri for found, _ in cflib.crtp.scan_interfaces(address)):
                print(f'{name} answers on {uri}', flush=True)
                del missing[name]
        if not missing:
            break
        if time.monotonic() > deadline:
            raise SystemExit('Not answering on the radio: '+', '.join(f'{n} ({u})' for n, u in missing.items())
                             +'. Power the drone, wait for it to boot, check the Crazyradio, then start again.')
        if time.monotonic()-last > 5.:
            print('Waiting for '+', '.join(missing)+' on the radio ...', flush=True); last = time.monotonic()
        time.sleep(1.)
    return wanted


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', required=True)
    parser.add_argument('--configuration', required=True)
    parser.add_argument('--capture-only', action='store_true')
    args = parser.parse_args()
    configuration = json.loads(args.configuration)
    if not args.capture_only:
        wait_for_radios(configuration['crazyflies'])
    if args.capture_only:
        configuration['motion_capture'] = capture_configuration(configuration)
    record = Path('/tmp')/f'aerowhip-real-{args.session}.json'
    os.environ['AEROWHIP_REAL_SESSION'] = args.session
    process = None

    def interrupt(signum, frame):
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signum)

    signal.signal(signal.SIGINT, interrupt)
    signal.signal(signal.SIGTERM, interrupt)
    with tempfile.TemporaryDirectory(prefix='aerowhip-real-') as directory:
        paths = {}
        for name in ('crazyflies', 'motion_capture'):
            path = Path(directory)/(name+'.yaml')
            path.write_text(configuration[name], encoding='utf-8')
            paths[name] = path
        process = subprocess.Popen(stack_command(paths, args.capture_only), start_new_session=True)
        try:
            with record.open('x') as handle:
                json.dump(dict(session=args.session, pid=process.pid), handle)
            print(f'{"Capture-only" if args.capture_only else "Real ROS"} launch PID {process.pid}; viewer uses read-only OptiTrack feedback.', flush=True)
            result = process.wait()
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGINT)
                process.wait()
            record.unlink(missing_ok=True)
    raise SystemExit(result)


if __name__ == '__main__':
    main()
