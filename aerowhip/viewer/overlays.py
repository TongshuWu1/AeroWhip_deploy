"""Reusable display actors for reference/trail paths and physical force arrows."""

from collections import deque
import numpy as np


COLORS = {'reference': '#008c82', 'trail': '#9653b5', 'tip_trail': '#dd517b',
          'thrust': '#287ad5', 'cable': '#e07925', 'command': '#e8b400'}


class SimulationOverlays:
    def __init__(self, plotter, pv, state, *, prefix=''):
        self.visible = dict.fromkeys(COLORS, True)
        self.force_scale = .25  # displayed metres per newton; no clipping/normalization
        self.acceleration_scale = .08  # displayed metres per m/s^2 (commanded acceleration arrow)
        self.trail = deque(maxlen=2000)
        self.tip_trail = deque(maxlen=2000)
        self.last_time = None
        self.trajectory_epoch = state.trajectory_epoch
        self.reference_count = 0
        self.lines, self.actors = {}, {}
        for key in ('reference', 'trail', 'tip_trail'):
            line = pv.lines_from_points(np.tile(state.drone.position_world_m, (2, 1)))
            self.lines[key] = line
            self.actors[key] = plotter.add_mesh(line, name=f'{prefix}motion_{key}',
                color=COLORS[key], line_width=2 if key == 'reference' else 3,
                lighting=False, pickable=False, reset_camera=False)
        for key in ('thrust', 'cable', 'command'):
            self.actors[key] = plotter.add_mesh(
                pv.Arrow(start=(0, 0, 0), direction=(1, 0, 0), scale=1.,
                         tip_length=.22, tip_radius=.055, shaft_radius=.018),
                name=f'{prefix}force_{key}', color=COLORS[key], lighting=False,
                pickable=False, reset_camera=False)
        for actor in self.actors.values():
            actor.SetVisibility(False)

    def reset_trail(self):
        self.trail.clear()
        self.tip_trail.clear()
        self.last_time = None
        self.actors['trail'].SetVisibility(False)
        self.actors['tip_trail'].SetVisibility(False)

    def set_reference(self, points):
        if points is None:
            self.reference_count = 0
        else:
            points = np.asarray(points, dtype=float)
            if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
                raise ValueError('Reference path must contain finite XYZ points.')
            indices = np.linspace(0, len(points)-1, min(len(points), 5000), dtype=int)
            points = points[indices]
            self.reference_count = len(points)
            if len(points) > 1:
                self._set_line('reference', points)
        self.actors['reference'].SetVisibility(self.visible['reference'] and self.reference_count > 1)

    def _set_line(self, key, points):
        # Actors and PolyData are retained; only coordinates/connectivity change.
        line = self.lines[key]
        line.points = np.asarray(points, dtype=float)
        line.lines = np.r_[len(points), np.arange(len(points))]

    def _arrow(self, key, origin, force, scale=None):
        scale = self.force_scale if scale is None else scale
        magnitude = float(np.linalg.norm(force))
        actor = self.actors[key]
        show = self.visible[key] and magnitude > 1e-10
        actor.SetVisibility(show)
        if not show:
            return
        direction = force/magnitude
        helper = np.array([0., 0., 1.]) if abs(direction[2]) < .9 else np.array([0., 1., 0.])
        side = np.cross(helper, direction)
        side /= np.linalg.norm(side)
        transform = np.eye(4)
        # Only length encodes magnitude. A fixed transverse width keeps the
        # smaller reaction arrow visible against the physical cable geometry.
        transform[:3, :3] = np.column_stack((direction, side, np.cross(direction, side))) @ np.diag(
            (magnitude*scale, .17, .17))
        transform[:3, 3] = origin
        actor.user_matrix = transform

    def update(self, state):
        if state.trajectory_epoch != self.trajectory_epoch:
            self.reset_trail()
            self.set_reference(None)
            self.trajectory_epoch = state.trajectory_epoch
        if self.last_time is not None and state.time_s < self.last_time:
            self.reset_trail()
        if self.last_time is None or state.time_s > self.last_time:
            self.trail.append(state.drone.position_world_m.copy())
            if state.cable_node_positions_world_m is not None:
                self.tip_trail.append(state.cable_node_positions_world_m[-1].copy())
            self.last_time = state.time_s
            if len(self.trail) > 1:
                self._set_line('trail', np.asarray(self.trail))
            if len(self.tip_trail) > 1:
                self._set_line('tip_trail', np.asarray(self.tip_trail))
        self.actors['trail'].SetVisibility(self.visible['trail'] and len(self.trail) > 1)
        self.actors['tip_trail'].SetVisibility(self.visible['tip_trail'] and len(self.tip_trail) > 1)
        self.actors['reference'].SetVisibility(self.visible['reference'] and self.reference_count > 1)
        command = getattr(state, 'commanded_acceleration_world_mps2', None)
        if command is None:
            self.actors['command'].SetVisibility(False)
        else:
            self._arrow('command', state.drone.position_world_m, command, self.acceleration_scale)
        if state.forces is None:
            for key in ('thrust', 'cable'):
                self.actors[key].SetVisibility(False)
        else:
            self._arrow('thrust', state.forces.center_of_mass_world_m, state.forces.thrust_world_n)
            self._arrow('cable', state.attachment_position_world_m, state.forces.cable_force_world_n)
