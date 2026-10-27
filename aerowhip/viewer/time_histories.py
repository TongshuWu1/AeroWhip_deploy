"""Rolling time histories of the selected robot for the flight viewer (simulation, replay, live).

Drone acceleration per axis, body tilt, cable-tip speed and, when the source provides model
forces, the cable force on the drone, over the last `window_s` seconds of state time. The
buffer restarts when state time goes backwards (reset or seek) or the robot changes.
"""
from collections import deque

import numpy as np

from aerowhip.ui.plotting import PlotCanvas
from aerowhip.ui.style import AXIS, PALETTE


class TimeHistories(PlotCanvas):
    def __init__(self, parent=None, *, window_s=10., dark=True):
        super().__init__(parent, dark=dark, minimum=(300, 160))
        self.window_s = window_s
        self.robot = None
        self.samples = deque()
        self.tip = None
        self.has_forces = None
        self._build(False)

    def _build(self, forces):
        self.has_forces = forces
        with self.style():
            self.reset()
            n = 4 if forces else 3
            axes = self.figure.subplots(1, n)
            self.axes = axes
            acc, tilt, tip = axes[:3]
            self.lines = {}
            for k in range(3):
                self.lines[f'a{k}'], = acc.plot([], [], color=AXIS[k], label='xyz'[k])
            acc.set_title('Drone acceleration (m/s²)')
            acc.legend(loc='upper left', ncols=3)
            self.lines['tilt'], = tilt.plot([], [], color=PALETTE['orange'])
            tilt.set_title('Tilt from vertical (deg)')
            self.lines['tip'], = tip.plot([], [], color=PALETTE['sky'])
            tip.set_title('Cable-tip speed (m/s)')
            if forces:
                self.lines['cable'], = axes[3].plot([], [], color=PALETTE['purple'], label='cable on drone')
                self.lines['thrust'], = axes[3].plot([], [], color=AXIS[1], label='thrust')
                axes[3].set_title('Force magnitude (N)')
                axes[3].legend(loc='upper left', ncols=2)
            for ax in axes:
                ax.set_xlabel('time (s)')
        self.draw_idle()

    def add(self, frame, robot_name):
        robot = next((r for r in frame.robots if r.name == robot_name), None)
        if robot is None:
            return
        t = float(robot.time_s)
        if robot_name != self.robot or (self.samples and t < self.samples[-1][0]-1e-9):
            self.robot = robot_name
            self.samples.clear()
            self.tip = None
        if self.samples and abs(t-self.samples[-1][0]) < 1e-9:
            return
        drone = robot.drone
        R = np.asarray(drone.rotation_body_to_world, float)
        tilt = float(np.degrees(np.arccos(np.clip(R[2, 2], -1, 1))))
        acceleration = drone.linear_acceleration_world_mps2
        acceleration = np.full(3, np.nan) if acceleration is None else np.asarray(acceleration, float)
        # Cable-tip speed from successive displayed tip positions (the display carries no velocities).
        tip = np.nan
        cable = robot.cable
        position = None if cable is None else np.asarray(cable.positions_world_m, float)[-1]
        if position is not None and self.tip is not None and t > self.tip[0]:
            tip = float(np.linalg.norm(position-self.tip[1])/(t-self.tip[0]))
        self.tip = None if position is None else (t, position)
        forces = robot.forces
        cable_force = thrust = np.nan
        if forces is not None:
            cable_force = float(np.linalg.norm(forces.cable_force_world_n))
            thrust = float(np.linalg.norm(forces.thrust_world_n))
        self.samples.append((t, *acceleration, tilt, tip, cable_force, thrust))
        while self.samples and self.samples[0][0] < t-self.window_s:
            self.samples.popleft()

    def redraw(self):
        if not self.isVisible() or not self.samples:
            return
        data = np.asarray(self.samples, float)
        forces = bool(np.isfinite(data[:, 6]).any())
        if forces != self.has_forces:
            self._build(forces)
        t = data[:, 0]
        available = bool(np.isfinite(data[:, 1:4]).any())
        self.axes[0].set_title('Drone acceleration (m/s²)' if available else
                               'Drone acceleration · not provided by this source')
        for k in range(3):
            self.lines[f'a{k}'].set_data(t, data[:, 1+k])
        self.lines['tilt'].set_data(t, data[:, 4])
        self.lines['tip'].set_data(t, data[:, 5])
        if forces:
            self.lines['cable'].set_data(t, data[:, 6])
            self.lines['thrust'].set_data(t, data[:, 7])
        right = t[-1]
        for ax in self.axes:
            ax.set_xlim(max(right-self.window_s, t[0]), max(right, t[0]+1e-3))
            ax.relim()
            ax.autoscale_view(scalex=False)
        self.draw_idle()
