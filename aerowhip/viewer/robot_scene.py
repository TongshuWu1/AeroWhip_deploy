"""Persistent CAD/cable actors for one named robot; entirely display-only."""
import numpy as np

from aerowhip.viewer.drone_model import add_drone_model
from aerowhip.viewer.overlays import SimulationOverlays


class RobotScene:
    def __init__(self, plotter, pv, state, settings):
        self.plotter, self.settings = plotter, settings
        self.name = state.name
        before = set(plotter.renderer.actors)
        self.drone_actors, self.rotors = add_drone_model(plotter, pv, state, settings, name=state.name)
        self.initial_position = state.drone.position_world_m.copy()
        self.initial_rotation = state.drone.rotation_body_to_world.copy()
        self.initial_world_to_body = np.eye(4)
        self.initial_world_to_body[:3, :3] = self.initial_rotation.T
        self.initial_world_to_body[:3, 3] = -self.initial_rotation.T @ self.initial_position
        self.body_axes = []
        labels = []
        for i, color in enumerate(('#ff6b6b', '#69db7c', '#74c0fc')):
            direction = self.initial_rotation[:, i]
            self.body_axes.append(plotter.add_mesh(pv.Arrow(start=self.initial_position,
                direction=direction, scale=settings.body_axis_length_m,
                tip_length=.22, tip_radius=.08, shaft_radius=.025),
                color=color, name=f'robot/{state.name}/axis/{i}', pickable=False))
            labels.append(self.initial_position+direction*settings.body_axis_length_m*1.08)
        self.body_labels = pv.PolyData(np.asarray(labels))
        self.initial_body_labels = self.body_labels.points.copy()
        self.body_axes.append(plotter.add_point_labels(self.body_labels, ['X_B', 'Y_B', 'Z_B'],
            name=f'robot/{state.name}/body_labels', show_points=False, always_visible=True,
            shape=None, font_size=10, text_color='#344958'))
        self.attachment_labels = pv.PolyData(state.attachment_position_world_m.reshape(1, 3).copy())
        self.labels = [plotter.add_point_labels(self.attachment_labels, [state.name],
            name=f'robot/{state.name}/name', show_points=False, always_visible=True,
            shape=None, font_size=11, text_color='#344958')]
        self.cable_actor = None
        self.marker_label_actor = None
        self.marker_actors = []
        self.topology = self.signature(state)
        if state.cable is not None:
            self._add_cable(plotter, pv, state)
        self.overlays = SimulationOverlays(plotter, pv, state, prefix=f'robot/{state.name}/')
        self.actors = [actor for key, actor in plotter.renderer.actors.items() if key not in before]
        self.update(state)

    @staticmethod
    def signature(state):
        if state.cable is None:
            return None
        return (len(state.cable.positions_world_m), tuple(state.cable.marker_indices))

    def _add_cable(self, plotter, pv, state):
        from vtkmodules.vtkFiltersCore import vtkTubeFilter
        positions = state.cable.positions_world_m
        self.cable_line = pv.lines_from_points(positions.copy(), close=False)
        self.cable_tube = vtkTubeFilter()
        self.cable_tube.SetInputData(self.cable_line)
        self.cable_tube.SetRadius(self.settings.cable_radius_m)
        self.cable_tube.SetNumberOfSides(18)
        self.cable_tube.CappingOn()
        self.cable_actor = plotter.add_mesh(self.cable_tube, name=f'robot/{state.name}/cable',
            color='#7bbd29', smooth_shading=True, pbr=True, metallic=0., roughness=.72)
        self.marker_indices = state.cable.marker_indices[state.cable.marker_indices != 0]
        self.initial_markers = positions[self.marker_indices].copy()
        for i, position in enumerate(self.initial_markers):
            self.marker_actors.append(plotter.add_mesh(pv.Sphere(radius=self.settings.cable_marker_radius_m,
                center=position, theta_resolution=20, phi_resolution=14),
                name=f'robot/{state.name}/marker/{i}', color='#eeeae0', smooth_shading=True,
                pbr=True, metallic=.08, roughness=.48))
        if len(self.initial_markers):
            self.marker_labels = pv.PolyData(self.initial_markers.copy())
            self.marker_label_actor = plotter.add_point_labels(self.marker_labels,
                [f'C{i}' for i in self.marker_indices], name=f'robot/{state.name}/marker_labels',
                show_points=False, always_visible=True, shape=None, font_size=10, text_color='#344958')
            self.labels.append(self.marker_label_actor)

    def update(self, state):
        rotation = state.drone.rotation_body_to_world @ self.initial_rotation.T
        translation = state.drone.position_world_m-rotation @ self.initial_position
        transform = np.eye(4)
        transform[:3, :3], transform[:3, 3] = rotation, translation
        for actor in self.drone_actors+self.body_axes[:-1]:
            actor.user_matrix = transform
        body_to_world = np.eye(4)
        body_to_world[:3, :3] = state.drone.rotation_body_to_world
        body_to_world[:3, 3] = state.drone.position_world_m
        phase = 2*np.pi*((state.time_s*self.settings.visual_rotor_rpm/60) % 1)
        for rotor in self.rotors:
            angle = rotor.direction*phase
            c, s = np.cos(angle), np.sin(angle)
            spin = np.eye(4)
            spin[:3, :3] = ((c, -s, 0), (s, c, 0), (0, 0, 1))
            center = np.asarray(rotor.center_body_m)
            spin[:3, 3] = center-spin[:3, :3] @ center
            for actor in rotor.actors:
                actor.user_matrix = body_to_world @ spin @ self.initial_world_to_body
        self.body_labels.points[:] = self.initial_body_labels @ rotation.T+translation
        self.attachment_labels.points[:] = state.attachment_position_world_m
        if self.cable_actor is not None:
            self.cable_line.points[:] = state.cable.positions_world_m
            markers = state.cable.positions_world_m[self.marker_indices]
            for actor, current, initial in zip(self.marker_actors, markers, self.initial_markers):
                actor.position = current-initial
            if len(markers):
                self.marker_labels.points[:] = markers
        self.overlays.update(state)

    def set_cable_opacity(self, opacity):
        for actor in ([self.cable_actor] if self.cable_actor is not None else [])+self.marker_actors:
            actor.GetProperty().SetOpacity(float(opacity))

    def visible(self, *, robot=True, cable=True, markers=True, body_axes=False, labels=False):
        for actor in self.drone_actors+[a for r in self.rotors for a in r.actors]:
            actor.SetVisibility(robot)
        for actor in self.body_axes:
            actor.SetVisibility(robot and body_axes)
        for actor in self.labels:
            actor.SetVisibility(robot and labels)
        if self.marker_label_actor is not None:
            self.marker_label_actor.SetVisibility(robot and cable and markers and labels)
        if self.cable_actor is not None:
            self.cable_actor.SetVisibility(robot and cable)
        for actor in self.marker_actors:
            actor.SetVisibility(robot and cable and markers)
        if not robot:
            for actor in self.overlays.actors.values():
                actor.SetVisibility(False)

    def remove(self):
        for actor in self.actors:
            self.plotter.remove_actor(actor, reset_camera=False, render=False)
