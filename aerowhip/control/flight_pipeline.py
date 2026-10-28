"""The flight pipeline for the interactive simulators: the host's tracking law, then the onboard
firmware (the native FlightController, as in training and fitting).

In flight the host turns a position/velocity/acceleration reference into an acceleration command
with a PID on the tracked position (the cf_2 flight scripts, `AccelerationController` in
crazyswarm2_jazzy_whipping/scripts/acceleration_circle_cf2.py: the gains below), adds the vertical
feed-forward that holds drone and cable, and sends it as the firmware's acceleration setpoint with the
onboard translation gains zeroed. `direct` skips the host law and passes the reference acceleration
through (a flight script that runs its own host law, as in the ROS rehearsal).

Commands are 13-vectors: position, velocity, acceleration (world, m, m/s, m/s^2, gravity excluded),
yaw (rad), body rates (unused by the firmware law; kept for the command format).
"""
import numpy as np

from aerowhip.fitting.flight_drone import firmware_config
from aerowhip.physics.model import native_module

KP = np.array([4., 4., 4.])
KD = np.array([3., 3., 3.])
KI = np.array([.6, .6, .6])
INTEGRAL_LIMIT = 1.5      # m/s^2 of integral action per axis


class FlightPipeline:
    def __init__(self, model, *, firmware=None, battery_voltage=None):
        self.model = model
        self.firmware = firmware_config() if firmware is None else firmware
        self.voltage = self.firmware['reference_voltage_v'] if battery_voltage is None else float(battery_voltage)
        self.flight = native_module().FlightController(self.firmware, model.config, self.voltage)
        self.controller_period_s = model.config['controller']['controller_period_s']
        self.direct = False       # or a callable answering it at each step (the ROS rehearsal)
        self.hover_trim = 0.
        self.integral = np.zeros(3)

    def prepare_reference(self, command, jerk=None):
        """A finite 11- or 13-vector reference as a 13-vector; body rates and jerk do not enter the
        firmware law and are dropped."""
        command = np.asarray(command, dtype=np.float64)
        if command.shape not in ((11,), (13,)) or not np.isfinite(command).all():
            raise ValueError('Expected a finite position/velocity/acceleration/yaw reference')
        result = np.zeros(13)
        result[:10] = command[:10]
        return result

    def settle(self, state, yaw):
        """Hover trim at this state (the vertical acceleration command whose delivered thrust holds
        drone and cable, as in training) and a cleared integral. Returns the hover thrust (N)."""
        weight = -(self.model.config['drone']['mass_kg']+self.model.config['cable']['total_mass_kg']) \
            * self.model.config['simulation']['gravity_world_m_s2'][2]
        low, high = -5., 12.
        for _ in range(60):
            mid = .5*(low+high)
            if self.flight(state, self.acceleration_command([0., 0., mid], yaw))[0] < weight:
                low = mid
            else:
                high = mid
        self.hover_trim, self.integral = .5*(low+high), np.zeros(3)
        self.flight.reset()
        return float(self.flight(state, self.acceleration_command([0., 0., self.hover_trim], yaw))[0])

    @staticmethod
    def acceleration_command(acceleration, yaw):
        command = np.zeros(13)
        command[6:9], command[9] = acceleration, yaw
        return command

    def command(self, position, velocity, reference, dt):
        """The command the host sends: the reference's position and velocity (for the record), the
        host's acceleration and the reference yaw."""
        if self.direct() if callable(self.direct) else self.direct:
            acceleration = reference[6:9]
        else:
            error = reference[:3]-position
            self.integral = np.clip(self.integral+error*dt, -INTEGRAL_LIMIT/KI, INTEGRAL_LIMIT/KI)
            acceleration = reference[6:9]+KP*error+KD*(reference[3:6]-velocity)+KI*self.integral
        command = self.acceleration_command(acceleration, reference[9])
        command[:6] = reference[:6]
        command[8] += self.hover_trim
        return command
