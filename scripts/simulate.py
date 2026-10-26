import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # Repository root: runs without installing.
from aerowhip.paths import ROOT
import argparse
import signal
from threading import Event

from aerowhip.physics.configuration import DEFAULT_MODEL, load_configuration
from aerowhip.physics.runtime import Simulation
from aerowhip.physics.controller_interface import ControllerReceiver, DEFAULT_PORT
from aerowhip.viewer.pyvista_viewer import PyVistaViewer
from aerowhip.control.gamepad.manual_control import GamepadCommandSource


DEFAULT_CONFIGURATION = ROOT / "configs" / "scenes" / "default.toml"
MODEL_CONFIGURATION = DEFAULT_MODEL  # configs/simulation.json links shared settings.


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the simulation independently of its controller.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIGURATION,
                        help="Scene configuration (TOML)")
    parser.add_argument("--model", type=Path, default=MODEL_CONFIGURATION,
                        help="Simulation setup or saved model snapshot (JSON)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args()
    configuration = load_configuration(args.config)
    manual = GamepadCommandSource(configuration.initial_state, configuration.timestep_s)
    simulation = Simulation(configuration, model_config=args.model,
                            command_source=manual, command_reset=manual.reset)
    receiver = ControllerReceiver(args.port, running=lambda: simulation.running)
    def poll_controls():
        controls = receiver.poll()
        manual.controls = controls
        return controls
    stop = Event()
    handlers = {sig: signal.signal(sig, lambda *_: stop.set())
                for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        viewer = PyVistaViewer(configuration)
        viewer.stop_requested = stop
        viewer.show(simulation, control_source=poll_controls)
    finally:
        receiver.close()
        for sig, handler in handlers.items():
            signal.signal(sig, handler)


if __name__ == "__main__":
    main()
