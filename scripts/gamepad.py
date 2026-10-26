"""Run the standalone virtual/Xbox controller; the simulator runs separately."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # Repository root: runs without installing.

import argparse

from aerowhip.physics.controller_interface import ControlInput, ControllerClient, DEFAULT_PORT
from aerowhip.control.gamepad.virtual_gamepad import VirtualGamepad
from aerowhip.control.gamepad.xbox_input import XboxGamepad


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args()
    panel = VirtualGamepad()
    xbox = XboxGamepad()
    client = ControllerClient(args.port)
    mode = "virtual"

    def tick():
        nonlocal mode
        if panel.closed:
            return
        virtual = panel.poll()
        if panel.mode != mode:
            mode = panel.mode
            xbox.disarm()
            virtual = ControlInput()
        if mode == "xbox":
            physical = xbox.poll()
            controls = ControlInput(physical.x, physical.y, physical.z, physical.yaw,
                                    physical.toggle or virtual.toggle, physical.reset or virtual.reset)
        else:
            controls = virtual
        if controls.reset:
            panel.clear()
            xbox.disarm()
            controls = ControlInput(reset=True)
        connected = client.send(controls)
        if not connected:
            panel.clear()
            xbox.disarm()
        state = ("Running" if client.running else "Paused") if connected else "Waiting for simulation"
        if connected and not client.armed:
            state = "Release controls"
        panel.status.set(state + ("\n" + xbox.status if mode == "xbox" else ""))
        panel.run_button.configure(text="Pause" if connected and client.running else "Run")
        panel.root.after(20, tick)

    panel.root.after(0, tick)
    try:
        panel.root.mainloop()
    finally:
        client.close()
        panel.close()


if __name__ == "__main__":
    main()
