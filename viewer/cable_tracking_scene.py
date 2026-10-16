"""Persistent raw-cloud and estimated cable actors, separate from physics."""
import numpy as np


class CableTrackingScene:
    def __init__(self, plotter, pv, name):
        self.plotter = plotter
        self.meshes = [pv.PolyData(np.zeros((1, 3))) for _ in range(3)]
        self.actors = []
        for mesh, suffix, color, size in zip(self.meshes, ('raw', 'nodes', 'links'),
                ('#9aa0ac', '#00a9e8', '#00a9e8'), (5, 11, 1)):
            self.actors.append(plotter.add_mesh(mesh, name=f'cable_tracking/{name}/{suffix}',
                color=color, point_size=size, line_width=3, render_points_as_spheres=True,
                lighting=False, pickable=False))

    def update(self, data, frame_time_s, *, visible=True, raw_visible=True):
        if data is None:
            for actor in self.actors:
                actor.SetVisibility(False)
            return
        raw, nodes, links = self.meshes
        raw.points = data.raw_points_world_m.copy()
        raw.verts = np.column_stack((np.ones(len(raw.points), dtype=int), np.arange(len(raw.points)))).ravel()
        nodes.points = data.positions_world_m[data.observed].copy()
        nodes.verts = np.column_stack((np.ones(len(nodes.points), dtype=int), np.arange(len(nodes.points)))).ravel()
        links.points = np.vstack((data.root_world_m, data.positions_world_m))
        mask = np.r_[True, data.observed]
        edges = np.flatnonzero(mask[:-1] & mask[1:])
        links.verts = np.empty(0, dtype=int)
        links.lines = np.column_stack((np.full(len(edges), 2), edges, edges+1)).ravel()
        fresh = frame_time_s-data.time_s <= .25
        for mesh, actor in zip(self.meshes, self.actors):
            actor.SetVisibility(visible and fresh and mesh.n_cells > 0)
        self.actors[0].SetVisibility(visible and raw_visible and fresh and raw.n_cells > 0)

    def remove(self):
        for actor in self.actors:
            self.plotter.remove_actor(actor, reset_camera=False, render=False)
