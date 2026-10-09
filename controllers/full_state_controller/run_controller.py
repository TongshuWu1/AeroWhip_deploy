"""Select a scene and CSV, then open the full-state controller and simulator."""

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

from simulation.runtime import Simulation
from simulation.configuration import DEFAULT_MODEL
from viewer.pyvista_viewer import PyVistaViewer
from viewer.simulation_window import SimulationWindow
from simulation.controller_interface import ControlInput
from PySide6 import QtWidgets
from controllers.full_state_controller.controller_window import ControllerWindow
from trajectories.full_state import COLUMNS
from controllers.full_state_controller.setup_window import SetupWindow


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/scenes/default.toml',
                        help='Scene configuration (TOML)')
    parser.add_argument('--model', type=Path, default=DEFAULT_MODEL,
                        help='Simulation setup or self-contained model snapshot (JSON)')
    parser.add_argument('--trajectory', type=Path,
                        help='Timestamped CSV (SI units): ' + ','.join(COLUMNS) +
                             '. Default: gentle 0.25 m X movement.')
    args = parser.parse_args()
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    try:
        setup = SetupWindow(ROOT, args.config, args.trajectory, args.model)
    except (OSError, ValueError, TypeError) as exc:
        QtWidgets.QMessageBox.critical(None, 'Cannot load setup', str(exc))
        return
    if setup.exec() != QtWidgets.QDialog.DialogCode.Accepted:
        return
    config, trajectory, source, model_config = setup.selection
    try:
        simulation = Simulation(config, model_config=model_config,
                                command_source=trajectory)
    except (OSError, ValueError, RuntimeError) as exc:
        QtWidgets.QMessageBox.critical(None, 'Cannot open simulation', str(exc))
        return
    viewer = PyVistaViewer(config)
    window = SimulationWindow(viewer, simulation, ControlInput)
    panel = ControllerWindow(viewer, simulation, trajectory, source)
    window.attach_controller(panel)
    if config.viewer.start_maximized:
        window.showMaximized()
    else:
        window.show()
    panel.show()
    try:
        app.exec()
    finally:
        panel.close()
        window.close()


if __name__ == '__main__':
    main()
