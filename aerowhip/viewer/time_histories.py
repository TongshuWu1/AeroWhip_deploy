"""Rolling time histories of the selected robot for the flight viewer (simulation, replay, live).

Top row: the drone's acceleration along world x, y and z, each with the desired (planned
reference) and the commanded (sent, trims and feedback included) acceleration when the source
provides them (real flights). Bottom row: body tilt, cable-tip speed and, when the source
provides model forces, the cable force on the drone. Over the last `window_s` seconds of state
time; the buffer restarts when state time goes backwards (reset or seek) or the robot changes.
"""
from collections import deque

import numpy as np

from aerowhip.ui.plotting import PlotCanvas
from aerowhip.ui.style import AXIS, PALETTE

# Columns of a sample row.
T, MEASURED, DESIRED, COMMANDED, TILT, TIP, CABLE, THRUST = 0, 1, 4, 7, 10, 11, 12, 13


class TimeHistories(PlotCanvas):
    def __init__(self, parent=None, *, window_s=10., dark=True):
        super().__init__(parent, dark=dark, minimum=(300, 240))
        self.window_s = window_s
        self.robot = None
        self.samples = deque()
        self.tip = None
        self._build()

    def _build(self):
        with self.style():
            self.reset()
            axes = self.figure.subplots(2, 3)
            self.axes = axes.ravel()
            self.lines = {}
            for k in range(3):
                ax = axes[0, k]
                self.lines[f'a{k}'], = ax.plot([], [], color=AXIS[k], label='measured')
                self.lines[f'd{k}'], = ax.plot([], [], color=PALETTE['sky'], linestyle='--', label='desired')
                self.lines[f'c{k}'], = ax.plot([], [], color=PALETTE['orange'], linestyle=':', label='commanded')
                ax.set_title(f'Acceleration {"xyz"[k]} (m/s²)')
            axes[0, 0].legend(loc='upper left', ncols=3, fontsize='small')
            tilt, tip, force = axes[1]
            self.lines['tilt'], = tilt.plot([], [], color=PALETTE['orange'])
            tilt.set_title('Tilt from vertical (deg)')
            self.lines['tip'], = tip.plot([], [], color=PALETTE['sky'])
            tip.set_title('Cable-tip speed (m/s)')
            self.lines['cable'], = force.plot([], [], color=PALETTE['purple'], label='cable on drone')
            self.lines['thrust'], = force.plot([], [], color=AXIS[1], label='thrust')
            force.set_title('Force magnitude (N)')
            force.legend(loc='upper left', ncols=2, fontsize='small')
            for ax in axes[1]:
                ax.set_xlabel('time (s)')
        self.draw_idle()

    @staticmethod
    def _vector(value):
        return np.full(3, np.nan) if value is None else np.asarray(value, float)

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
        measured = self._vector(drone.linear_acceleration_world_mps2)
        desired = self._vector(getattr(robot, 'desired_acceleration_world_mps2', None))
        commanded = self._vector(getattr(robot, 'commanded_acceleration_world_mps2', None))
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
        self.samples.append((t, *measured, *desired, *commanded, tilt, tip, cable_force, thrust))
        while self.samples and self.samples[0][0] < t-self.window_s:
            self.samples.popleft()

    def redraw(self):
        if not self.isVisible() or not self.samples:
            return
        data = np.asarray(self.samples, float)
        t = data[:, T]
        for k in range(3):
            self.lines[f'a{k}'].set_data(t, data[:, MEASURED+k])
            self.lines[f'd{k}'].set_data(t, data[:, DESIRED+k])
            self.lines[f'c{k}'].set_data(t, data[:, COMMANDED+k])
        if not np.isfinite(data[:, MEASURED:MEASURED+3]).any():
            self.axes[0].set_title('Acceleration x · not provided by this source', loc='left',
                                   fontsize=self.rc['axes.titlesize'])
        self.lines['tilt'].set_data(t, data[:, TILT])
        self.lines['tip'].set_data(t, data[:, TIP])
        self.lines['cable'].set_data(t, data[:, CABLE])
        self.lines['thrust'].set_data(t, data[:, THRUST])
        right = t[-1]
        for ax in self.axes:
            ax.set_xlim(max(right-self.window_s, t[0]), max(right, t[0]+1e-3))
            ax.relim()
            ax.autoscale_view(scalex=False)
        self.draw_idle()
