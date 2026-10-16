"""Select Crazyswarm metadata without treating firmware settings as plant gains."""
import json
import re
import subprocess

from deployment.ros_bridge import vector


def read_yaml_in_container(container, paths):
    # PyYAML is already in ROS Docker; the host simulation needs no ROS/YAML install.
    documents = [path.read_text(encoding='utf-8') for path in paths]
    return parse_yaml_in_container(container, documents)


def parse_yaml_in_container(container, documents):
    """Parse the exact captured bytes also passed to a managed real launch."""
    command = ['docker', 'exec', '-i', container, 'python3', '-c',
        'import json,sys,yaml; json.dump([yaml.safe_load(x) for x in json.load(sys.stdin)],sys.stdout,allow_nan=False)']
    result = subprocess.run(command, input=json.dumps(documents), capture_output=True, text=True, check=True)
    return json.loads(result.stdout)


def select_robot(base, overlay=None, name=None):
    if not isinstance(base, dict) or not isinstance(base.get('robots'), dict):
        raise ValueError('Crazyflies YAML must contain a robots mapping')
    types = base.get('robot_types', {})
    enabled = []
    for key, data in base['robots'].items():
        if data.get('enabled') is True and types.get(data.get('type'), {}).get('connection', 'crazyflie') == 'crazyflie':
            enabled.append(key)
    if name is None:
        if len(enabled) != 1:
            raise ValueError(f'This adapter owns one drone/cable plant; select --robot from enabled robots: {enabled}')
        name = enabled[0]
    if name not in enabled or not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', name):
        raise ValueError(f'Robot {name!r} must be an enabled Crazyflie with a valid ROS name')
    data = base['robots'][name]
    frame = 'world'
    if base.get('fileversion', 0) >= 3:
        for scope in (base.get('all', {}), types.get(data.get('type'), {}), data):
            frame = scope.get('reference_frame', frame)
    if frame != 'world':
        raise ValueError('The current PVA adapter supports the world reference frame only')
    override = (overlay or {}).get('robots', {}).get(name, {})
    if set(override) - {'initial_position'}:
        raise ValueError('Simulation YAML robot overrides support initial_position only')
    original = vector(data.get('initial_position'), 3)
    initial = vector(override.get('initial_position', original), 3)
    uri = data.get('uri')
    if not isinstance(uri, str) or not re.fullmatch(r'.*[0-9A-Fa-f]{2}', uri):
        raise ValueError('Crazyflie URI must have a hexadecimal ID suffix')
    return dict(name=name, uri=uri, type=data.get('type', ''), frame=frame,
        initial_position=initial, yaml_initial_position=original,
        airborne_landing_height_m=initial[2], enabled_robots=enabled,
        configured_firmware_params=base.get('all', {}).get('firmware_params', {}))
