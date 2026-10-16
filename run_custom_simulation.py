"""Launch native free-space simulation with the Crazyswarm ROS command interface."""
import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import signal
import subprocess
import time
from threading import Event
import uuid

from deployment.custom_simulation import (DomainInUseError, DomainLease, Evidence, PortInUseError,
    RosSimulationSource, SimulationTransport, launch_container_script)
from deployment.ros_bridge import DEFAULT_DOMAIN, DEFAULT_PORT
from deployment.configuration import read_yaml_in_container, select_robot
from deployment.flight_commands import FlightCommands
from deployment.stack_lifecycle import capture_simulation_bridge, close_captured_stack
import numpy as np
from simulation.configuration import DEFAULT_MODEL, load_configuration
from simulation.runtime import Simulation
from project_paths import project_path

ROOT = Path(__file__).resolve().parent
DEFAULT_CONTAINER = 'crazyswarm2_jazzy_whipping_dev'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--container', default=DEFAULT_CONTAINER)
    parser.add_argument('--robot', help='Select one enabled robot; inferred when exactly one is enabled')
    parser.add_argument('--crazyflies-yaml', type=Path,
        default=ROOT.parent/'Crazyflie/crazyswarm2_jazzy_whipping/config/crazyflies.yaml')
    parser.add_argument('--simulation-yaml', type=Path,
        help='Optional explicit initial-position overrides; defaults to the Crazyflies YAML position')
    parser.add_argument('--domain', type=int, default=DEFAULT_DOMAIN)
    parser.add_argument('--port', type=int, default=DEFAULT_PORT)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/scenes/default.toml')
    parser.add_argument('--model', type=Path, default=DEFAULT_MODEL)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--duration', type=float, help='Headless wall-time limit in seconds')
    args = parser.parse_args()
    if args.domain == 42 or not 0 <= args.domain <= 232:
        parser.error('Choose a simulation ROS domain in 0..232, separate from real-flight domain 42')
    if args.robot is not None and not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', args.robot):
        parser.error('Robot name must be a nonempty ROS name without slashes')
    if args.duration is not None and (not args.headless or not math.isfinite(args.duration) or args.duration <= 0):
        parser.error('--duration requires --headless and a positive duration')
    if not 0 <= args.port <= 65535:
        parser.error('Port must be in 0..65535 (zero selects an unused port)')
    args.config, args.model = project_path(args.config), project_path(args.model)
    inspection = subprocess.run(['docker','inspect','--format','{{.State.Running}}',args.container], check=True, capture_output=True, text=True)
    if inspection.stdout.strip() != 'true':
        parser.error(f'{args.container} is stopped; start it with ./docker/compose in the Crazyswarm project')
    try:
        yaml_paths = [args.crazyflies_yaml.resolve()]
        if args.simulation_yaml is not None:
            yaml_paths.append(args.simulation_yaml.resolve())
        documents = read_yaml_in_container(args.container, yaml_paths)
        robot = select_robot(documents[0], documents[1] if len(documents) == 2 else None, args.robot)
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        parser.error(f'Crazyflies configuration: {exc}')
    args.robot = robot['name']
    session = uuid.uuid4().hex
    output = project_path(args.output) if args.output else ROOT/'output/custom_simulation'/f'{datetime.now(timezone.utc):%Y%m%d-%H%M%S}-{session[:8]}'
    evidence = Evidence(output)
    source = transport = process = log = domain_lease = bridge_target = None
    stop = Event()
    handlers = {sig: signal.signal(sig, lambda *_: stop.set())
                for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        domain_lease = DomainLease(args.domain)
        configuration = load_configuration(args.config)
        initial = configuration.initial_state
        position = np.asarray(robot['initial_position'], dtype=np.float64)
        delta = position-initial.position_world_m
        cable = initial.cable_node_positions_world_m
        initial = replace(initial, position_world_m=position,
            cable_node_positions_world_m=None if cable is None else cable+delta)
        configuration = replace(configuration, initial_state=initial)
        # The viewer's floor is display geometry, not a physical boundary.
        # Keep the configured world coordinates even below that visual plane;
        # model.initial_state still validates the physical cable constraints.
        commands = FlightCommands(args.robot, evidence)
        simulation = Simulation(configuration, model_config=args.model, command_source=commands, command_reset=commands.reset)
        source = RosSimulationSource(simulation, commands, session, evidence)
        transport = SimulationTransport(source, port=args.port, domain=args.domain, evidence=evidence)
        metadata = dict(session=session, domain=args.domain, robot=args.robot, host_port=transport.port,
            scene=str(args.config.resolve()), model=str(args.model.resolve()), container=args.container,
            model_configuration=simulation.dynamics.model.config, initial_position_world_m=configuration.initial_state.position_world_m.tolist(),
            robot_configuration=robot, yaml_sources=[dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in yaml_paths],
            status='Connection evidence, not firmware equivalence or real-flight validation',
            adapter_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (ROOT/'deployment').glob('*.py')})
        with (output/'metadata.json').open('x') as file:
            file.write(json.dumps(metadata, indent=2, allow_nan=False)+'\n')
        log = (output/'ros-bridge.log').open('x')
        process = launch_container_script(args.container, ROOT/'deployment/ros_bridge.py',
            ['--robot',args.robot,'--port',transport.port,'--session',session,
             '--robot-config',json.dumps(robot, allow_nan=False)], domain=args.domain, log=log,
             start_new_session=True)
        if not transport.connected.wait(10.):
            raise RuntimeError(f'ROS bridge did not connect; inspect {output/"ros-bridge.log"}')
        bridge_target = capture_simulation_bridge(args.container, args.domain, session)
        if bridge_target is None:
            raise RuntimeError('Could not identify the owned simulation ROS bridge')
        if stop.is_set():
            return
        source.running = True
        print(f'Simulation flight: {args.robot} · ROS domain {args.domain} · localhost:{transport.port}', flush=True)
        print(f'Command/feedback evidence: {output}', flush=True)
        print(f'YAML position: {robot["yaml_initial_position"]}; simulation initial position: {robot["initial_position"]}', flush=True)
        print('Free-space dynamics: the displayed floor has no contact response. Disarm holds position; emergency suspends physics.', flush=True)
        if args.headless:
            start = time.monotonic()
            while not stop.wait(.05) and (args.duration is None or time.monotonic()-start < args.duration):
                if source.latest.error or process.poll() is not None:
                    raise RuntimeError(source.latest.error or 'ROS adapter exited; inspect ros-bridge.log')
        else:
            from viewer.pyvista_viewer import PyVistaViewer
            viewer = PyVistaViewer(configuration)
            viewer.stop_requested = stop
            viewer.show_source(source)
    except KeyboardInterrupt:
        pass
    except (PortInUseError, DomainInUseError) as exc:
        parser.exit(status=1, message=f'Custom simulation: {exc}\n')
    finally:
        # Tell the ROS child to exit before stopping its native owner.
        try:
            if transport:
                transport.close()
        finally:
            try:
                if source:
                    source.close()
            finally:
                try:
                    if process:
                        if bridge_target is None:
                            bridge_target = capture_simulation_bridge(args.container, args.domain, session)
                        close_captured_stack(args.container, args.domain, bridge_target)
                        process.wait(5.)
                finally:
                    if log:
                        log.close()
                    if domain_lease:
                        domain_lease.close()
                    evidence.close()
                    for sig, handler in handlers.items():
                        signal.signal(sig, handler)


if __name__ == '__main__':
    main()
