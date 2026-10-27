"""Time histories of one saved drone-cable motion (simulation or replay arrays).

Panels, sharing time: commanded vs achieved drone acceleration per axis; tilt implied by
the command vs achieved tilt; cable-tip speed and its component along the task's output
direction; motor thrusts. Achieved acceleration is the finite difference of the saved
drone velocity. Panels whose arrays were not saved are omitted.
"""
import numpy as np

from aerowhip.ui.plotting import PlotCanvas, message
from aerowhip.ui.style import AXIS, PALETTE, ROLE

GRAVITY = 9.81


def tilt_deg(rotations):
    return np.degrees(np.arccos(np.clip(rotations[:, 2, 2], -1, 1)))


def command_tilt_deg(acceleration):
    """Tilt of the thrust direction that realises a world acceleration (gravity excluded)."""
    return np.degrees(np.arctan2(np.linalg.norm(acceleration[:, :2], axis=1), GRAVITY+acceleration[:, 2]))


class MotionPlots(PlotCanvas):
    def __init__(self, parent=None, *, minimum=(360, 420)):
        super().__init__(parent, minimum=minimum)

    def set_arrays(self, arrays, output_direction=None):
        a = arrays
        t = np.asarray(a['time_s'], float)
        panels = []
        if 'applied_command' in a and 'drone_velocities' in a:
            panels += ['acceleration', 'tilt'] if 'drone_rotations' in a else ['acceleration']
        if 'cable_velocities' in a:
            panels.append('tip')
        if 'motor_thrust_n' in a:
            panels.append('motors')
        with self.style():
            self.reset()
            if not panels:
                message(self.figure.add_subplot(), 'No time histories saved with this motion')
                self.draw_idle()
                return
            axes = np.atleast_1d(self.figure.subplots(len(panels), 1, sharex=True))
            for ax, panel in zip(axes, panels):
                if panel == 'acceleration':
                    command = np.asarray(a['applied_command'], float)[:, :3]
                    achieved = np.gradient(np.asarray(a['drone_velocities'], float), t, axis=0)
                    for k in range(3):
                        ax.plot(t, command[:, k], color=AXIS[k], linewidth=1.0, label=f"{'xyz'[k]} command")
                        ax.plot(t, achieved[:, k], color=AXIS[k], linewidth=.9, linestyle=':', alpha=.9)
                    ax.plot([], [], color=ROLE['measured'], linestyle=':', label='achieved (dotted)')
                    ax.set_ylabel('a (m/s²)')
                    ax.legend(loc='upper left', ncols=4)
                elif panel == 'tilt':
                    command = np.asarray(a['applied_command'], float)[:, :3]
                    ax.plot(t, command_tilt_deg(command), color=ROLE['command'], label='commanded')
                    ax.plot(t, tilt_deg(np.asarray(a['drone_rotations'], float)), color=ROLE['measured'],
                            label='achieved')
                    ax.set_ylabel('tilt (°)')
                    ax.set_ylim(bottom=0)
                    ax.legend(loc='upper left', ncols=2)
                elif panel == 'tip':
                    tip = np.asarray(a['cable_velocities'], float)[:, -1]
                    ax.plot(t, np.linalg.norm(tip, axis=1), color=ROLE['model'], label='|v|')
                    if output_direction is not None:
                        ax.plot(t, tip@np.asarray(output_direction, float), color=PALETTE['orange'],
                                label='along output dir.')
                    ax.axhline(0, color=ROLE['reference'], linewidth=.6)
                    ax.set_ylabel('tip (m/s)')
                    ax.legend(loc='upper left', ncols=2)
                elif panel == 'motors':
                    thrust = np.asarray(a['motor_thrust_n'], float)
                    for k in range(thrust.shape[1]):
                        ax.plot(t, thrust[:, k], linewidth=.9, label=f'm{k+1}')
                    ax.set_ylabel('motor (N)')
                    ax.legend(loc='upper left', ncols=4)
            axes[-1].set_xlabel('time (s)')
            axes[-1].set_xlim(t[0], t[-1])
            for ax in axes:
                self.add_cursor(ax, t[0])
        self.draw_idle()
