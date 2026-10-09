from __future__ import annotations

from time import perf_counter

import numpy as np

from simulation.configuration import SimulationConfiguration
from simulation.runtime import Simulation
from simulation.controller_interface import ControlInput
from simulation.state import SimulationState
from viewer.drone_model import add_drone_model
from viewer.environment import add_environment, configure_lighting
from viewer.window import configure_native_window, enable_native_resolution
from viewer.overlays import SimulationOverlays


class PyVistaViewer:
    def __init__(self, configuration: SimulationConfiguration) -> None:
        self._configuration = configuration
        self._plotter: object | None = None
        self._body_axis_actors: list[object] = []
        self._world_axis_actors: list[object] = []
        self._label_actors: list[object] = []
        self._body_axes_visible = False
        self._world_axes_visible = True
        self._labels_visible = False
        self._simulation: Simulation | None = None
        self._dynamics_error: str | None = None
        self._control_source = ControlInput
        self._window = None
        self._overlays = None
        self._reference_points = None
        self._playback_speed = 1.

    def create_plotter(self, state: SimulationState, *, off_screen: bool = False, plotter=None):
        """Build the same scene for interactive viewing and off-screen inspection."""
        embedded = plotter is not None
        if not off_screen and not embedded:
            enable_native_resolution()
        try:
            import pyvista as pv
        except ImportError as exc:
            raise RuntimeError(
                "PyVista is required to run the simulation viewer."
            ) from exc

        viewer = self._configuration.viewer
        if plotter is None:
            plotter = pv.Plotter(window_size=viewer.window_size_px, off_screen=off_screen)
        self._plotter = plotter
        self._label_actors = []
        # MSAA smooths geometric edges without filtering the entire text/image.
        plotter.enable_anti_aliasing("msaa", multi_samples=4)
        configure_lighting(plotter, pv, viewer)

        add_environment(plotter, pv, viewer)
        self._add_target(plotter, pv)
        self._world_axis_actors = self._add_world_axes(plotter, pv)
        self._add_drone(plotter, pv, state)
        self._add_cable(plotter, pv, state)
        self._overlays = SimulationOverlays(plotter, pv, state)
        self._overlays.set_reference(self._reference_points)
        self._update_geometry(state)
        self._set_visibility(self._body_axis_actors, self._body_axes_visible)
        self._set_visibility(self._world_axis_actors, self._world_axes_visible)
        self._set_visibility(self._label_actors, self._labels_visible)
        if not embedded:
            self._register_key_controls(plotter)
        center = 0.5 * (state.drone.position_world_m + state.cable_node_positions_world_m[-1])
        camera_offset = np.array((1.45, -2.0, 0.90))
        if state.forces is not None:
            # Frame initial force endpoints once; never move the user's camera
            # automatically during simulation or when an arrow changes.
            points = np.vstack((state.cable_node_positions_world_m, state.drone.position_world_m))
            span = max(float(np.linalg.norm(np.ptp(points, axis=0))), .1)
            points = np.vstack((points,
                state.forces.center_of_mass_world_m + self._overlays.force_scale*state.forces.thrust_world_n,
                state.attachment_position_world_m + self._overlays.force_scale*state.forces.cable_force_world_n))
            center = .5*(points.min(axis=0)+points.max(axis=0))
            camera_offset *= max(1., float(np.linalg.norm(np.ptp(points, axis=0)))/span)
        if viewer.target_position_world_m is not None:
            # Frame the target along with the drone/cable; leave ordinary scenes unchanged.
            target_distance = np.linalg.norm(viewer.target_position_world_m - center)
            center = 0.5 * (center + viewer.target_position_world_m)
            camera_offset *= max(1.0, target_distance / 1.25)
        self._initial_camera = [
            center + camera_offset, center, (0.0, 0.0, 1.0)
        ]
        plotter.camera_position = self._initial_camera
        if not embedded:
            configure_native_window(plotter, start_maximized=viewer.start_maximized)
        else:
            # Deferred display-only controls for saved-state windows; keep
            # playback-layout indices unchanged during their construction.
            from viewer.replay_trajectories import schedule_saved_trajectories
            schedule_saved_trajectories(plotter, self)
        return plotter

    def show(self, simulation: Simulation, *, control_source=ControlInput) -> None:
        from viewer.simulation_window import show_simulation
        show_simulation(self, simulation, control_source)

    def _add_target(self, plotter: object, pv: object) -> None:
        position = self._configuration.viewer.target_position_world_m
        if position is None:
            return
        # Three world-aligned strokes; their size is visual, not a hit tolerance.
        for axis, offset in enumerate(np.eye(3) * 0.06):
            plotter.add_mesh(
                pv.Line(position - offset, position + offset),
                name=f"target_{axis}", color="#ed790b", line_width=4,
                lighting=False, pickable=False,
            )

    def _on_timer(self, *_args) -> None:
        now = perf_counter()
        elapsed = now - self._last_tick
        self._last_tick = now
        if self._simulation is not None:
            try:
                controls = self._control_source()
                if controls.reset:
                    self._reset_simulation()
                    return
                if controls.toggle:
                    self._toggle_simulation()
                    elapsed = 0.0
                changed = self._simulation.advance(elapsed*self._playback_speed)
            except Exception as exc:
                self._simulation.running = False
                self._dynamics_error = str(exc)
                self._update_status()
                return
            if changed:
                self._update_geometry(self._simulation.state)
                self._plotter.render()

    def _update_status(self) -> None:
        if self._simulation is None or self._plotter is None:
            return
        if self._window is not None:
            self._window.update_telemetry()
        self._plotter.render()

    def _toggle_simulation(self) -> None:
        if self._simulation is not None and self._dynamics_error is None:
            self._simulation.toggle_running()
            self._last_tick = perf_counter()
            self._update_status()

    def _step_simulation(self) -> None:
        if self._simulation is not None and self._dynamics_error is None:
            try:
                self._simulation.step()
            except Exception as exc:
                self._dynamics_error = str(exc)
            else:
                self._update_geometry(self._simulation.state)
            self._last_tick = perf_counter()
            self._update_status()

    def _reset_simulation(self) -> None:
        if self._simulation is not None:
            self._simulation.reset()
            self._dynamics_error = None
            self._last_tick = perf_counter()
            if self._overlays is not None:
                self._overlays.reset_trail()
            self._update_geometry(self._simulation.state)
            self._update_status()

    def set_trajectory(self, trajectory) -> None:
        self._reference_points = None if trajectory is None else trajectory.preview('Position')
        if self._overlays is not None:
            self._overlays.set_reference(self._reference_points)
        if self._plotter is not None:
            self._plotter.render()

    def set_playback_speed(self, speed) -> None:
        if speed not in (.25, .5, 1.):
            raise ValueError('Playback speed must be 0.25, 0.5 or 1.')
        self._playback_speed = float(speed)
        self._last_tick = perf_counter()

    def set_overlay_visible(self, name, visible) -> None:
        if self._overlays is None:
            return
        self._overlays.visible[name] = bool(visible)
        sim = self._simulation
        if name in ('thrust', 'cable') and hasattr(sim, 'set_force_telemetry'):
            sim.set_force_telemetry(any(self._overlays.visible[key] for key in ('thrust', 'cable')))
        if sim is not None:
            self._overlays.update(sim.state)
        self._update_status()

    def set_force_scale(self, scale) -> None:
        if self._overlays is not None:
            self._overlays.force_scale = float(scale)
            if self._simulation is not None:
                self._overlays.update(self._simulation.state)
            self._update_status()

    def _add_world_axes(self, plotter: object, pv: object) -> list[object]:
        length = self._configuration.viewer.world_axis_length_m
        origin = np.zeros(3, dtype=np.float64)
        actors: list[object] = []
        axes = (
            (np.array([1.0, 0.0, 0.0]), "#e74c3c", "X_W"),
            (np.array([0.0, 1.0, 0.0]), "#2ecc71", "Y_W"),
            (np.array([0.0, 0.0, 1.0]), "#3498db", "Z_W"),
        )
        label_points = []
        labels = []
        for direction, color, label in axes:
            arrow = pv.Arrow(
                start=origin,
                direction=direction,
                scale=length,
                tip_length=0.22,
                tip_radius=0.08,
                shaft_radius=0.025,
            )
            actors.append(plotter.add_mesh(arrow, color=color))
            label_points.append(direction * length * 1.08)
            labels.append(label)

        actors.append(
            plotter.add_point_labels(
                np.asarray(label_points),
                labels,
                show_points=False,
                always_visible=False,
                shape=None,
                font_size=11,
                text_color="#344958",
            )
        )
        return actors

    def _add_drone(
        self, plotter: object, pv: object, state: SimulationState
    ) -> None:
        viewer = self._configuration.viewer
        position = state.drone.position_world_m
        rotation = state.drone.rotation_body_to_world
        self._drone_actors, self._rotors = add_drone_model(plotter, pv, state, viewer)
        self._initial_drone_position = position.copy()
        self._initial_drone_rotation = rotation.copy()
        self._initial_world_to_body = np.eye(4)
        self._initial_world_to_body[:3, :3] = rotation.T
        self._initial_world_to_body[:3, 3] = -rotation.T @ position
        self._body_axis_actors = self._add_body_axes(
            plotter, pv, position, rotation
        )
        self._attachment_label_points = pv.PolyData(state.attachment_position_world_m.reshape(1, 3).copy())
        self._label_actors.append(
            plotter.add_point_labels(
                self._attachment_label_points,
                ["attachment"],
                show_points=False,
                always_visible=True,
                shape=None,
                font_size=10,
                text_color="#ee9b32",
            )
        )

    def _add_body_axes(
        self,
        plotter: object,
        pv: object,
        origin: np.ndarray,
        rotation_body_to_world: np.ndarray,
    ) -> list[object]:
        length = self._configuration.viewer.body_axis_length_m
        colors = ("#ff6b6b", "#69db7c", "#74c0fc")
        labels = ("X_B", "Y_B", "Z_B")
        actors: list[object] = []
        label_points = []
        for axis_index, color in enumerate(colors):
            direction = rotation_body_to_world[:, axis_index]
            arrow = pv.Arrow(
                start=origin,
                direction=direction,
                scale=length,
                tip_length=0.22,
                tip_radius=0.08,
                shaft_radius=0.025,
            )
            actors.append(plotter.add_mesh(arrow, color=color))
            label_points.append(origin + direction * length * 1.08)

        self._body_label_points = pv.PolyData(np.asarray(label_points))
        self._initial_body_label_points = self._body_label_points.points.copy()
        actors.append(
            plotter.add_point_labels(
                self._body_label_points,
                list(labels),
                show_points=False,
                always_visible=True,
                shape=None,
                font_size=10,
                text_color="#344958",
            )
        )
        return actors

    def _add_cable(
        self, plotter: object, pv: object, state: SimulationState
    ) -> None:
        viewer = self._configuration.viewer
        positions = state.cable_node_positions_world_m
        from vtkmodules.vtkFiltersCore import vtkTubeFilter
        self._cable_line = pv.lines_from_points(positions.copy(), close=False)
        self._cable_tube = vtkTubeFilter()
        self._cable_tube.SetInputData(self._cable_line)
        self._cable_tube.SetRadius(viewer.cable_radius_m)
        self._cable_tube.SetNumberOfSides(18)
        self._cable_tube.CappingOn()
        plotter.add_mesh(
            self._cable_tube,
            color="#7bbd29",
            smooth_shading=True,
            pbr=True,
            metallic=0.0,
            roughness=0.72,
        )

        marker_positions = positions[state.cable_marker_indices[1:]]
        self._initial_marker_positions = marker_positions.copy()
        self._marker_actors = []
        for marker_position in marker_positions:
            marker = pv.Sphere(
                radius=viewer.cable_marker_radius_m,
                center=marker_position,
                theta_resolution=20,
                phi_resolution=14,
            )
            self._marker_actors.append(plotter.add_mesh(
                marker,
                color="#eeeae0",
                smooth_shading=True,
                pbr=True,
                metallic=0.08,
                roughness=0.48,
            ))

        self._marker_label_points = pv.PolyData(marker_positions.copy())
        self._label_actors.append(
            plotter.add_point_labels(
                self._marker_label_points,
                [
                    f"C{index}"
                    for index in range(1, marker_positions.shape[0] + 1)
                ],
                show_points=False,
                always_visible=True,
                shape=None,
                font_size=10,
                text_color="#344958",
            )
        )

    def _update_geometry(self, state: SimulationState) -> None:
        """Reuse actor/mesh identities; transfer only the current displayed state."""
        rotation = state.drone.rotation_body_to_world @ self._initial_drone_rotation.T
        translation = state.drone.position_world_m - rotation @ self._initial_drone_position
        transform = np.eye(4)
        transform[:3, :3] = rotation
        transform[:3, 3] = translation
        for actor in self._drone_actors + self._body_axis_actors[:-1]:
            actor.user_matrix = transform
        body_to_world = np.eye(4)
        body_to_world[:3, :3] = state.drone.rotation_body_to_world
        body_to_world[:3, 3] = state.drone.position_world_m
        # Simulation time makes pause, single-step and reset deterministic.
        # This visual phase is not used by the dynamics or a thrust model.
        phase = 2 * np.pi * ((state.time_s * self._configuration.viewer.visual_rotor_rpm / 60) % 1)
        for rotor in self._rotors:
            angle = rotor.direction * phase
            c, s = np.cos(angle), np.sin(angle)
            spin = np.eye(4)
            spin[:3, :3] = ((c, -s, 0), (s, c, 0), (0, 0, 1))
            center = np.asarray(rotor.center_body_m)
            spin[:3, 3] = center - spin[:3, :3] @ center
            rotor_transform = body_to_world @ spin @ self._initial_world_to_body
            for actor in rotor.actors:
                actor.user_matrix = rotor_transform
        self._body_label_points.points[:] = self._initial_body_label_points @ rotation.T + translation
        self._attachment_label_points.points[:] = state.attachment_position_world_m
        self._cable_line.points[:] = state.cable_node_positions_world_m
        markers = state.cable_node_positions_world_m[state.cable_marker_indices[1:]]
        for actor, current, initial in zip(self._marker_actors, markers, self._initial_marker_positions):
            actor.position = current - initial
        self._marker_label_points.points[:] = markers
        if self._overlays is not None:
            self._overlays.update(state)

    def _register_key_controls(self, plotter: object) -> None:
        # Handle both KeyPressEvent and VTK's separate CharEvent: otherwise W
        # also enables wireframe and R overrides our saved camera position.
        for key, callback in (("b", self._toggle_body_axes),
                              ("w", self._toggle_world_axes),
                              ("l", self._toggle_labels),
                              ("r", self._reset_camera),
                              ("i", self._toggle_info),
                              ("space", self._toggle_simulation),
                              ("n", self._step_simulation),
                              ("t", self._reset_simulation)):
            plotter.clear_events_for_key(key)
            plotter.add_key_event(key, callback)

        def handle_character(style, _event):
            if plotter.iren.interactor.GetKeySym() not in (
                "b", "w", "l", "r", "i", "space", "n", "t"
            ):
                style.OnChar()

        plotter.iren.style.AddObserver("CharEvent", handle_character)

    def _toggle_info(self) -> None:
        if self._window is not None:
            self._window.sidebar.setVisible(not self._window.sidebar.isVisible())

    def _reset_camera(self) -> None:
        if self._plotter is not None:
            self._plotter.camera_position = self._initial_camera
            self._plotter.render()

    def _toggle_body_axes(self) -> None:
        self._body_axes_visible = not self._body_axes_visible
        self._set_visibility(self._body_axis_actors, self._body_axes_visible)

    def _toggle_world_axes(self) -> None:
        self._world_axes_visible = not self._world_axes_visible
        self._set_visibility(self._world_axis_actors, self._world_axes_visible)

    def _toggle_labels(self) -> None:
        self._labels_visible = not self._labels_visible
        self._set_visibility(self._label_actors, self._labels_visible)

    def _set_visibility(self, actors: list[object], visible: bool) -> None:
        for actor in actors:
            actor.SetVisibility(visible)
        if self._plotter is not None:
            self._plotter.render()
