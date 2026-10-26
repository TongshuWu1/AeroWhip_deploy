"""ROS-only controller selection; no firmware parameter emulation or plant edits."""
import numpy as np
from aerowhip.control.geometric_controller import GeometricController


class RosController:
    def __init__(self, model, commands):
        self.commands = commands
        self.pva = GeometricController(model)
        self.direct = GeometricController(model)
        # The same nominal attitude/wrench calculation and static cable-weight
        # trim as the training acceleration path; P/V fields have no effect.
        self.direct.kp = np.zeros(3)
        self.direct.kv = np.zeros(3)
        self.controller_period_s = self.pva.controller_period_s

    def prepare_reference(self, command, jerk=None):
        return self.pva.prepare_reference(command, jerk)

    def set_command(self, command):
        self.pva.set_command(command)
        self.direct.set_command(command)

    def __call__(self, state):
        return (self.direct if self.commands.acceleration_only and self.commands.armed else self.pva)(state)
