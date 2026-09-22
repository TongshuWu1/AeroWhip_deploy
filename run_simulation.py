from pathlib import Path
import argparse
import sys


ROOT = Path(__file__).resolve().parent
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

from simulation.configuration import DEFAULT_MODEL, load_configuration
from simulation.runtime import Simulation
from simulation.controller_interface import ControllerReceiver, DEFAULT_PORT
from viewer.pyvista_viewer import PyVistaViewer
from controllers.gamepad_controller.manual_control import GamepadCommandSource


DEFAULT_CONFIGURATION = ROOT / "configs" / "scenes" / "default.toml"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the simulation independently of its controller.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIGURATION,
                        help="Scene configuration (TOML)")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL,
                        help="Dynamics model configuration (JSON)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--device", choices=("cpu", "cuda"),
                        help="Physics device (default: CUDA when available)")
    parser.add_argument("--no-compile", action="store_true", help="Use the uncompiled physics reference")
    args = parser.parse_args()
    configuration = load_configuration(args.config)
    manual = GamepadCommandSource(configuration.initial_state, configuration.timestep_s)
    simulation = Simulation(configuration, model_config=args.model, device=args.device,
                            compile_physics=not args.no_compile,
                            command_source=manual, command_reset=manual.reset)
    receiver = ControllerReceiver(args.port, running=lambda: simulation.running)
    def poll_controls():
        controls = receiver.poll()
        manual.controls = controls
        return controls
    try:
        PyVistaViewer(configuration).show(simulation, control_source=poll_controls)
    finally:
        receiver.close()


if __name__ == "__main__":
    main()
