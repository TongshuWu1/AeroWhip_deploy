"""The 3D view of the Identification Loop: the lab floor, the flight area (config/arena.yaml of the
flight stack) and ceiling, the planned path coloured by speed, and the drone (CAD frame) with the
simulated cable played back from a preview (scripts/preview_trajectory.py). Display only.

World frame as the simulator's: the hover point at (0, 0, 1.3 m + lift) above the floor; the
flight area is drawn centred on the hover point (where the drone is put down decides the real one).
"""
import tomllib

import numpy as np
import pyvista as pv
from pyvistaqt import QtInteractor
from PySide6 import QtWidgets

from aerowhip.paths import ROOT
from aerowhip.planning.simulate import FLIGHT_STACK
from aerowhip.viewer.drone_model import load_frame_meshes

SCENE = ROOT/'configs'/'scenes'/'default.toml'
LAB_CONFIG = ROOT/'configs'/'lab_scan.yaml'
BACKGROUND = ('#11161c', '#1f2933')
FLOOR, GRID, GRID_MAJOR = '#1a2129', '#26313c', '#34424f'
AREA, CEILING = '#3d93d1', '#D55E00'
CABLE, TIP, MARKER = '#f2f2f2', '#E69F00', '#56B4E9'
DRONE_BODY, DRONE_GUARD = '#2b2f36', '#c8d0d8'
PATH, AHEAD = '#93a0ad', '#3dd6c6'
FLOWN, FLOWN_CABLE = '#ffffff', '#ff5a4f'          # the measured flight: drone and its live-labelled cable        # the whole plan (faint) and the next seconds of it


def arena():
    import yaml
    try:
        return yaml.safe_load((FLIGHT_STACK/'config'/'arena.yaml').read_text())
    except OSError:
        return dict(x_m=[-2., 2.], y_m=[-1.5, 1.5], margin_m=.3)


def lab_scan_path():
    """The cleaned lab scan (scripts/clean_lab_scan.py), or None."""
    import yaml
    try:
        path = ROOT/yaml.safe_load(LAB_CONFIG.read_text())['cleaned']
    except (OSError, KeyError, TypeError):
        return None
    return path if path.exists() else None


def add_lab(plotter, path, name='lab'):
    """The lab scan, textured as scanned (its lighting is baked in), in OptiTrack coordinates."""
    import trimesh
    from PIL import Image
    scene = trimesh.load(path)
    actors = []
    for i, (key, g) in enumerate(scene.geometry.items()):
        T, _ = scene.graph[scene.graph.geometry_nodes[key][0]]
        g = g.copy()
        g.apply_transform(T)
        mesh = pv.wrap(g)
        material = getattr(g.visual, 'material', None)
        image = getattr(material, 'baseColorTexture', None) if material is not None else None
        if image is not None and getattr(g.visual, 'uv', None) is not None:
            mesh.active_texture_coordinates = np.asarray(g.visual.uv, np.float32)
            texture = pv.numpy_to_texture(np.asarray(image.convert('RGB') if isinstance(image, Image.Image) else image))
            # back faces culled: a wall between the camera and the room disappears (a doll's-house view)
            actors.append(plotter.add_mesh(mesh, texture=texture, lighting=False, culling='back', name=f'{name}/{i}', pickable=False))
        else:
            actors.append(plotter.add_mesh(mesh, color='#8c96a0', name=f'{name}/{i}', pickable=False))
    return actors


def viewer_settings():
    raw = tomllib.loads(SCENE.read_text())['viewer']
    return type('Viewer', (), raw)


class Scene3D(QtWidgets.QWidget):
    CEILING_M = 2.6               # highest planned height of the tracking origin (flight script)

    def __init__(self, parent=None, plotter=None):
        """plotter: an off-screen pyvista Plotter instead of the Qt one (for rendering images)."""
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        if plotter is None:
            plotter = QtInteractor(self, auto_update=False, multi_samples=8)
            layout.addWidget(plotter.interactor)
        self.plotter = plotter
        p = self.plotter
        p.set_background(BACKGROUND[0], top=BACKGROUND[1])
        p.enable_anti_aliasing('ssaa')
        p.add_light(pv.Light(position=(-3, -4, 7), focal_point=(0, 0, 1), intensity=.9, light_type='scene light'))
        p.add_light(pv.Light(position=(4, 2, 5), focal_point=(0, 0, 1), intensity=.5, light_type='scene light'))
        self.hover = np.array([0., 0., 1.3])
        self.preview = None
        self.flown = None
        self.trail = None
        self._drone = []
        self._frame_index = 0
        self._room()
        path = lab_scan_path()
        self.lab = add_lab(p, path) if path is not None else []
        self.set_lab_visible(bool(self.lab), render=False)
        self._add_drone()
        self.cable_line = pv.lines_from_points(np.tile(self.hover, (11, 1))-np.linspace(0, 1, 11)[:, None]*[0, 0, 1])
        self.cable_actor = p.add_mesh(self.cable_line.tube(radius=.004), color=CABLE, smooth_shading=True, name='cable')
        self.markers = pv.PolyData(np.zeros((10, 3)))
        p.add_mesh(self.markers, color=MARKER, point_size=9, render_points_as_spheres=True, name='markers')
        self.tip = pv.PolyData(np.zeros((1, 3)))
        p.add_mesh(self.tip, color=TIP, point_size=14, render_points_as_spheres=True, name='tip')
        p.add_axes(color='#93a0ad', line_width=2, labels_off=False)
        self.view('iso')

    # static scene
    def _room(self):
        p = self.plotter
        self.floor_actors = [p.add_mesh(pv.Plane(center=(0, 0, -.001), direction=(0, 0, 1), i_size=12, j_size=12), color=FLOOR,
                                        name='floor', ambient=.4, diffuse=.6)]
        for spacing, colour, name in ((.5, GRID, 'grid'), (1., GRID_MAJOR, 'grid_major')):
            lines = []
            for v in np.arange(-5, 5+1e-9, spacing):
                lines += [[(v, -5, 0), (v, 5, 0)], [(-5, v, 0), (5, v, 0)]]
            pts = np.array(lines).reshape(-1, 3)
            cells = np.hstack([[2, 2*i, 2*i+1] for i in range(len(lines))])
            self.floor_actors.append(p.add_mesh(pv.PolyData(pts, lines=cells), color=colour, line_width=1, name=name, lighting=False))
        a = arena()
        cx, cy = self.hover[0]-np.mean(a['x_m']), self.hover[1]-np.mean(a['y_m'])
        x0, x1 = a['x_m'][0]+cx, a['x_m'][1]+cx
        y0, y1 = a['y_m'][0]+cy, a['y_m'][1]+cy
        box = pv.Box(bounds=(x0, x1, y0, y1, 0, self.CEILING_M))
        p.add_mesh(box.extract_feature_edges(), color=AREA, line_width=1.5, opacity=.55, name='area_edges', lighting=False)
        m = a['margin_m']
        inner = pv.Box(bounds=(x0+m, x1-m, y0+m, y1-m, 0, self.CEILING_M))
        p.add_mesh(inner.extract_feature_edges(), color=AREA, line_width=1, opacity=.22, name='area_margin', lighting=False)
        p.add_mesh(pv.Plane(center=((x0+x1)/2, (y0+y1)/2, .002), direction=(0, 0, 1), i_size=x1-x0, j_size=y1-y0),
                   color=AREA, opacity=.07, name='area_floor', lighting=False)
        p.add_mesh(pv.Plane(center=((x0+x1)/2, (y0+y1)/2, self.CEILING_M), direction=(0, 0, 1), i_size=x1-x0, j_size=y1-y0),
                   color=CEILING, opacity=.05, name='ceiling', lighting=False)

    def _add_drone(self):
        settings = viewer_settings()
        body, guard = load_frame_meshes(pv, np.array([0., 0., -.055]), settings)
        self._drone = [self.plotter.add_mesh(body, color=DRONE_BODY, smooth_shading=True, specular=.3, name='drone_body'),
                       self.plotter.add_mesh(guard, color=DRONE_GUARD, smooth_shading=True, specular=.4, name='drone_guard')]

    # content
    def set_trajectory(self, data, preview):
        """data: the trajectory file (relative positions, speeds); preview: the npz arrays or None."""
        p = self.plotter
        for name in ('path', 'trail', 'ahead', 'hover_point', 'hover_drop'):
            p.remove_actor(name, render=False)
        self.preview = preview
        self.plan = None
        if preview is not None:
            self.hover = np.asarray(preview['reference'][0], float)
        M = np.eye(4); M[:2, 3] = self.hover[:2]           # the scan is in OptiTrack coordinates: take-off spot at the origin
        for actor in self.lab:
            actor.user_matrix = M
        p.add_mesh(pv.Sphere(radius=.02, center=self.hover), color='#93a0ad', name='hover_point', lighting=False)
        p.add_mesh(pv.Line(self.hover, (self.hover[0], self.hover[1], 0)), color='#93a0ad', line_width=1, opacity=.35,
                   name='hover_drop', lighting=False)
        if data is not None:
            self.plan = (np.asarray(data['t']), np.asarray(data['position'])+self.hover)
            p.add_mesh(pv.lines_from_points(self.plan[1][::2]), color=PATH, opacity=.18, line_width=1, name='path', lighting=False)
            self.ahead = pv.lines_from_points(self.plan[1][:3])
            p.add_mesh(self.ahead, color=AHEAD, line_width=4, name='ahead', lighting=False, render_lines_as_tubes=True)
        if preview is not None:
            self.trail = pv.lines_from_points(preview['nodes'][:2, -1])
            p.add_mesh(self.trail, color=TIP, opacity=.7, line_width=2, name='trail', lighting=False)
            self.set_frame(0)
        else:
            self._place_drone(self.hover, np.eye(3), np.vstack([self.hover+[0, 0, -.055-.1*k] for k in range(11)]), None)
            self.plotter.render()

    def set_lab_visible(self, visible, render=True):
        """The scanned lab in place of the plain floor and grid (when there is a scan)."""
        visible = visible and bool(self.lab)
        for actor in self.lab:
            actor.SetVisibility(visible)
        for actor in self.floor_actors:
            actor.SetVisibility(not visible)
        if render:
            self.plotter.render()

    def frames(self):
        return 0 if self.preview is None else len(self.preview['t'])

    def set_frame(self, i, trail_s=1.5, ahead_s=3.):
        if self.preview is None:
            return
        pr = self.preview
        i = int(np.clip(i, 0, len(pr['t'])-1))
        self._frame_index = i
        nodes = pr['nodes'][i]
        self._place_drone(pr['position'][i], pr['rotation'][i], nodes, pr['marker_indices'])
        k = int(trail_s/max(pr['t'][1]-pr['t'][0], 1e-3))
        tip = pr['nodes'][max(0, i-k):i+1, -1]
        if len(tip) > 1:
            self.trail.copy_from(pv.lines_from_points(tip))
        if self.plan is not None:
            t, pos = self.plan
            j0, j1 = np.searchsorted(t, [pr['t'][i], pr['t'][i]+ahead_s])
            segment = pos[min(j0, len(t)-2):max(j1, j0+2)]
            self.ahead.copy_from(pv.lines_from_points(segment))
        if self.flown is not None:
            f = self.flown
            j = int(np.clip(np.searchsorted(f['t'], pr['t'][i]), 0, len(f['t'])-1))
            self.flown_drone.points = f['position'][j:j+1]
            seen = np.isfinite(f['markers'][j]).all(1)
            self.flown_markers.points = f['markers'][j][seen] if seen.any() else f['position'][j:j+1]
        self.plotter.render()

    def set_flown(self, record):
        """The measured flight on the same timeline (white: drone path and drone; red: the cable markers
        the live labeller sees), or None to hide it."""
        for name in ('flown_path', 'flown_drone', 'flown_markers'):
            self.plotter.remove_actor(name, render=False)
        self.flown = record
        if record is None:
            return
        inside = (record['t'] >= -1.) & (record['t'] <= (self.preview['t'][-1] if self.preview is not None else 1e9))
        self.plotter.add_mesh(pv.lines_from_points(record['position'][inside][::2]), color=FLOWN, line_width=1.5, opacity=.55,
                              name='flown_path', lighting=False)
        self.flown_drone = pv.PolyData(record['position'][:1])
        self.plotter.add_mesh(self.flown_drone, color=FLOWN, point_size=12, render_points_as_spheres=True, name='flown_drone')
        self.flown_markers = pv.PolyData(record['position'][:1])
        self.plotter.add_mesh(self.flown_markers, color=FLOWN_CABLE, point_size=10, render_points_as_spheres=True, name='flown_markers')
        if self.preview is not None:
            self.set_frame(self._frame_index)

    def _place_drone(self, position, R, nodes, marker_indices):
        M = np.eye(4); M[:3, :3] = R; M[:3, 3] = position
        for actor in self._drone:
            actor.user_matrix = M
        line = pv.lines_from_points(nodes)
        self.cable_actor.mapper.SetInputData(line.tube(radius=.004, n_sides=14))
        if marker_indices is not None:
            self.markers.points = nodes[np.asarray(marker_indices)[1:]] if len(marker_indices) > 1 else nodes[-1:]
        self.tip.points = nodes[-1:]

    def view(self, which):
        p = self.plotter
        focus = self.hover-[0, 0, .45]
        if which == 'iso':
            p.camera_position = [tuple(focus+[2.5, -2.9, 1.5]), tuple(focus), (0, 0, 1)]
        elif which == 'top':
            p.camera_position = [tuple(focus+[0, 0, 4.2]), tuple(focus), (0, 1, 0)]
        elif which == 'side':
            p.camera_position = [tuple(focus+[0, -3.4, .2]), tuple(focus), (0, 0, 1)]
        elif which == 'front':
            p.camera_position = [tuple(focus+[3.4, 0, .2]), tuple(focus), (0, 0, 1)]
        elif which == 'close' and self.preview is not None:
            d = self.preview['position'][self._frame_index]
            p.camera_position = [tuple(d+[1.2, -1.4, .5]), tuple(d-[0, 0, .4]), (0, 0, 1)]
        p.camera.view_angle = 32
        p.render()
