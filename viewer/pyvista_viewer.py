from __future__ import annotations

from time import perf_counter
import numpy as np

from simulation.configuration import SimulationConfiguration
from simulation.controller_interface import ControlInput
from simulation.state import SimulationState
from viewer.display_state import DisplayFrame, WorldDisplay, simulation_frame
from viewer.environment import add_environment, configure_lighting
from viewer.robot_scene import RobotScene
from viewer.window import configure_native_window, enable_native_resolution


class PyVistaViewer:
    def __init__(self, configuration: SimulationConfiguration, *, world=None) -> None:
        self._configuration = configuration
        self.world = world or WorldDisplay.from_configuration(configuration.viewer)
        self._plotter = self._window = self._simulation = self._overlays = None
        self._robot_scenes = {}
        self._tracking_scenes = {}
        self._robot_visibility = {}
        self._overlay_visibility = {}
        self._display_frame = None
        self._body_axis_actors, self._world_axis_actors, self._label_actors = [], [], []
        self._body_axes_visible, self._world_axes_visible, self._labels_visible = False, True, False
        self._world_visible = dict(floor=True, grid=True)
        self._dynamics_error = None
        self._reference_points = None
        self._playback_speed = 1.
        self._force_scale = .25
        self._last_revision = -1
        self._last_render = 0.
        self.follow_robot = None
        self._follow_position = None

    def _frame(self, state):
        frame = simulation_frame(state) if isinstance(state, SimulationState) else state
        if not isinstance(frame, DisplayFrame) or frame.frame_id != self.world.frame_id:
            raise ValueError('Display state must use the configured world frame; transforms belong to the source')
        return frame

    def create_plotter(self, state, *, off_screen=False, plotter=None):
        """Build persistent actors from a display frame (legacy states adapt here)."""
        frame = self._frame(state)
        embedded = plotter is not None
        if not off_screen and not embedded:
            enable_native_resolution()
        import pyvista as pv
        settings = self._configuration.viewer
        if plotter is None:
            plotter = pv.Plotter(window_size=settings.window_size_px, off_screen=off_screen)
        self._plotter = plotter
        self._robot_scenes = {}
        self._tracking_scenes = {}
        plotter.enable_anti_aliasing('msaa', multi_samples=4)
        configure_lighting(plotter, pv, self.world)
        add_environment(plotter, pv, self.world)
        self._add_target(plotter, pv)
        self._world_axis_actors = self._add_world_axes(plotter, pv)
        plotter.add_axes(interactive=False, color='#344958', line_width=2,
                         viewport=(0., 0., .14, .14))
        self._update_geometry(frame)
        self.camera_view('Orbit')
        self._initial_camera = list(plotter.camera_position)
        if not embedded:
            self._register_key_controls(plotter)
            configure_native_window(plotter, start_maximized=settings.start_maximized)
        else:
            from viewer.replay_trajectories import schedule_saved_trajectories
            schedule_saved_trajectories(plotter, self)
        return plotter

    def show(self, simulation, *, control_source=ControlInput):
        from viewer.simulation_window import show_simulation
        show_simulation(self, simulation, control_source)

    def show_replay(self, replay):
        from viewer.simulation_window import show_replay
        show_replay(self, replay)

    def show_source(self, source):
        from viewer.simulation_window import show_source
        show_source(self, source)

    def _add_target(self, plotter, pv):
        position = self._configuration.viewer.target_position_world_m
        if position is not None:
            for axis, offset in enumerate(np.eye(3)*.06):
                plotter.add_mesh(pv.Line(position-offset, position+offset), name=f'target_{axis}',
                    color='#ed790b', line_width=4, lighting=False, pickable=False)

    def _add_world_axes(self, plotter, pv):
        length = self.world.world_axis_length_m
        actors, points = [], []
        for axis, color in zip(np.eye(3), ('#e74c3c', '#2ecc71', '#3498db')):
            actors.append(plotter.add_mesh(pv.Arrow(start=(0, 0, 0), direction=axis, scale=length,
                tip_length=.22, tip_radius=.08, shaft_radius=.025), color=color, pickable=False))
            points.append(axis*length*1.08)
        actors.append(plotter.add_point_labels(np.asarray(points), ['X_W', 'Y_W', 'Z_W'],
            show_points=False, always_visible=False, shape=None, font_size=11, text_color='#344958'))
        return actors

    def _update_geometry(self, state):
        frame = self._frame(state)
        import pyvista as pv
        names = {r.name for r in frame.robots}
        for name in list(self._robot_scenes):
            if name not in names:
                self._robot_scenes.pop(name).remove()
        for name in list(self._tracking_scenes):
            if name not in names:
                self._tracking_scenes.pop(name).remove()
        for robot in frame.robots:
            scene = self._robot_scenes.get(robot.name)
            if scene is not None and scene.topology != RobotScene.signature(robot):
                scene.remove()
                scene = None
            if scene is None:
                scene = RobotScene(self._plotter, pv, robot, self._configuration.viewer)
                scene.overlays.force_scale = self._force_scale
                self._robot_scenes[robot.name] = scene
                for key, visible in self._overlay_visibility.items():
                    scene.overlays.visible[key] = visible
                if robot is frame.robots[0]:
                    scene.overlays.set_reference(self._reference_points)
            else:
                scene.update(robot)
            if robot.cable_tracking is not None and robot.name not in self._tracking_scenes:
                from viewer.cable_tracking_scene import CableTrackingScene
                self._tracking_scenes[robot.name] = CableTrackingScene(self._plotter, pv, robot.name)
            if robot.name in self._tracking_scenes:
                flags = self._robot_visibility.get(robot.name, {})
                self._tracking_scenes[robot.name].update(robot.cable_tracking, frame.time_s,
                    visible=flags.get('robot', True) and flags.get('tracking', True),
                    raw_visible=flags.get('raw_cloud', True))
            if self.follow_robot == robot.name:
                position = robot.drone.position_world_m
                if self._follow_position is not None:
                    delta = position-self._follow_position
                    camera = self._plotter.camera
                    camera.position = np.asarray(camera.position)+delta
                    camera.focal_point = np.asarray(camera.focal_point)+delta
                self._follow_position = position.copy()
        if self.follow_robot not in names:
            self._follow_position = None
        self._display_frame = frame
        # Kept for the existing embedded replay and path controls.
        primary = self._robot_scenes[frame.robots[0].name] if frame.robots else None
        self._overlays = None if primary is None else primary.overlays
        self._drone_actors = [] if primary is None else primary.drone_actors
        self._marker_actors = [] if primary is None else primary.marker_actors
        self._body_axis_actors = [] if primary is None else primary.body_axes
        self._label_actors = [] if primary is None else primary.labels
        if primary is not None and primary.cable_actor is not None:
            self._cable_line = primary.cable_line
        self._apply_visibility()

    def _apply_visibility(self):
        for key, actor in self._plotter.renderer.actors.items():
            if key.startswith('environment/floor'):
                actor.SetVisibility(self._world_visible['floor'])
            elif key.startswith('environment/grid'):
                actor.SetVisibility(self._world_visible['grid'])
        for actor in self._world_axis_actors:
            actor.SetVisibility(self._world_axes_visible)
        if self._plotter.renderer.axes_widget is not None:
            self._plotter.renderer.axes_widget.SetEnabled(self._world_axes_visible)
        for name, scene in self._robot_scenes.items():
            flags = dict(robot=True, cable=True, markers=True,
                         body_axes=self._body_axes_visible, labels=self._labels_visible)
            flags.update(self._robot_visibility.get(name, {}))
            flags.pop('tracking', None)
            flags.pop('raw_cloud', None)
            scene.visible(**flags)

    def set_display_visible(self, key, visible, *, robot=None):
        if robot is not None:
            self._robot_visibility.setdefault(robot, {})[key] = bool(visible)
        elif key in self._world_visible:
            self._world_visible[key] = bool(visible)
        elif key == 'world_axes':
            self._world_axes_visible = bool(visible)
        elif key == 'body_axes':
            self._body_axes_visible = bool(visible)
            for flags in self._robot_visibility.values():
                flags.pop('body_axes', None)
        elif key == 'labels':
            self._labels_visible = bool(visible)
        else:
            raise ValueError('Unknown display: '+key)
        if self._plotter is not None:
            if self._display_frame is not None:
                self._update_geometry(self._display_frame)
            else:
                self._apply_visibility()
            self._plotter.render()
        if self._window is not None:
            self._window.sync_display_checks()

    def set_overlay_visible(self, name, visible):
        self._overlay_visibility[name] = bool(visible)
        for scene in self._robot_scenes.values():
            scene.overlays.visible[name] = bool(visible)
        source = self._simulation
        if name in ('thrust', 'cable') and hasattr(source, 'set_force_telemetry'):
            self._action(lambda: source.set_force_telemetry(any(
                self._overlay_visibility.get(key, False) for key in ('thrust', 'cable'))))
        elif self._display_frame is not None:
            self._update_geometry(self._display_frame)
        self._update_status()

    def set_force_scale(self, scale):
        self._force_scale = float(scale)
        for scene in self._robot_scenes.values():
            scene.overlays.force_scale = float(scale)
        if self._display_frame is not None:
            self._update_geometry(self._display_frame)
        self._update_status()

    def set_trajectory(self, trajectory):
        self._reference_points = None if trajectory is None else trajectory.preview('Position')
        if self._overlays is not None:
            self._overlays.set_reference(self._reference_points)
        if self._plotter is not None:
            self._plotter.render()

    def set_playback_speed(self, speed):
        if speed not in (.25, .5, 1.):
            raise ValueError('Playback speed must be 0.25, 0.5 or 1')
        self._playback_speed = float(speed)
        if self._simulation is not None:
            self._action(lambda: self._simulation.set_speed(speed))
        self._last_tick = perf_counter()

    def _refresh_source(self, snapshot=None):
        if self._simulation is None:
            return
        snapshot = snapshot or self._simulation.latest
        if snapshot.revision != self._last_revision:
            self._last_revision = snapshot.revision
            try:
                self._update_geometry(snapshot.frame)
                self._plotter.render()
            except Exception as exc:
                self._dynamics_error = 'Display failed: '+str(exc)
                if self._simulation.has_playback and snapshot.running:
                    self._simulation.running = False
                if self._window is not None:
                    self._window.update_telemetry()
                return
            self._dynamics_error = snapshot.error
            self._last_render = perf_counter()

    def _on_timer(self, *_):
        now = perf_counter()
        elapsed, self._last_tick = now-self._last_tick, now
        if self._simulation is not None:
            snapshot = self._simulation.tick(elapsed)
            if now-self._last_render >= 1/60:
                self._refresh_source(snapshot)

    def _update_status(self):
        if self._window is not None:
            self._window.update_telemetry()
        if self._plotter is not None:
            self._plotter.render()

    def _action(self, action):
        try:
            action()
        except Exception as exc:
            self._dynamics_error = str(exc)
            self._update_status()
            return
        self._last_tick = perf_counter()
        self._refresh_source()
        self._update_status()

    def _toggle_simulation(self):
        if self._simulation is not None and self._simulation.has_playback:
            self._action(self._simulation.toggle_running)

    def _step_simulation(self):
        if self._simulation is not None and self._simulation.has_playback:
            self._action(self._simulation.step)

    def _reset_simulation(self):
        if self._simulation is not None and self._simulation.has_playback:
            for scene in self._robot_scenes.values():
                scene.overlays.reset_trail()
            self._action(self._simulation.reset)

    def clear_trails(self):
        """Clear display history only; never reset physics or send commands."""
        self._reference_points = None
        for scene in self._robot_scenes.values():
            scene.overlays.reset_trail()
            scene.overlays.set_reference(None)
        if self._plotter is not None:
            self._plotter.render()

    def _toggle_info(self):
        if self._window is not None:
            self._window.sidebar.setVisible(not self._window.sidebar.isVisible())

    def _reset_camera(self):
        self.camera_view('Orbit')
        if self._window is not None:
            from PySide6 import QtCore
            with QtCore.QSignalBlocker(self._window.camera):
                self._window.camera.setCurrentText('Orbit')

    def camera_view(self, view, *, robot=None):
        if self._plotter is None:
            return
        points = []
        for state in (() if self._display_frame is None else self._display_frame.robots):
            if robot is not None and state.name != robot:
                continue
            points.extend(state.drone.position_world_m+np.array([[-.13, -.13, -.08], [.13, .13, .08]]))
            if robot is None and state.cable is not None:
                points.extend(state.cable.positions_world_m)
            if robot is None and state.cable_tracking is not None:
                points.extend(state.cable_tracking.positions_world_m[state.cable_tracking.observed])
        target = self._configuration.viewer.target_position_world_m
        if target is not None and robot is None:
            points.append(target)
        points = np.asarray(points if points else [(0, 0, 1)], dtype=float)
        center = .5*(points.min(0)+points.max(0))
        span = max(.35, float(np.linalg.norm(np.ptp(points, axis=0))))
        direction, up = {
            'Orbit': ((1.45, -2., .9), (0, 0, 1)), 'Top': ((0, 0, 1), (0, 1, 0)),
            'Front': ((0, -1, 0), (0, 0, 1)), 'Side': ((1, 0, 0), (0, 0, 1)),
        }[view]
        direction = np.asarray(direction)/np.linalg.norm(direction)
        self._plotter.camera_position = [center+direction*span*2.5, center, up]
        self._plotter.camera.parallel_projection = view != 'Orbit'
        self._plotter.camera.parallel_scale = span*.65
        self._plotter.reset_camera_clipping_range()
        self._plotter.render()

    def set_follow(self, name):
        self.follow_robot = name
        self._follow_position = None

    def _toggle_body_axes(self):
        self.set_display_visible('body_axes', not self._body_axes_visible)

    def _toggle_world_axes(self):
        self.set_display_visible('world_axes', not self._world_axes_visible)

    def _toggle_labels(self):
        self.set_display_visible('labels', not self._labels_visible)

    def _register_key_controls(self, plotter):
        for key, callback in (('b', self._toggle_body_axes), ('w', self._toggle_world_axes),
                              ('l', self._toggle_labels), ('r', self._reset_camera),
                              ('i', self._toggle_info), ('space', self._toggle_simulation),
                              ('n', self._step_simulation), ('t', self._reset_simulation)):
            plotter.clear_events_for_key(key)
            plotter.add_key_event(key, callback)
        def handle_character(style, _event):
            if plotter.iren.interactor.GetKeySym() not in ('b', 'w', 'l', 'r', 'i', 'space', 'n', 't'):
                style.OnChar()
        plotter.iren.style.AddObserver('CharEvent', handle_character)
