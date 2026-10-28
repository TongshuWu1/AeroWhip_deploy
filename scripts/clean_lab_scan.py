"""Clean a phone scan of the lab (Polycam GLB) into the backdrop of the Identification Loop's 3D view.
Display only. Settings in configs/lab_scan.yaml.

    python scripts/clean_lab_scan.py              # writes the cleaned GLB and prints a summary
    python scripts/clean_lab_scan.py --render DIR # also renders top and perspective images

Steps: glTF Y-up to Z-up; the floor plane found by RANSAC and levelled to z = 0; faces outside
the room polygon (plus margin) or above the cut height removed; loose pieces of scan noise
removed; floor vertices snapped flat; then moved into OptiTrack coordinates (origin and +x as in
the config). Textures are kept as scanned.
"""
import argparse
from pathlib import Path
import sys

import numpy as np
import trimesh
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT/'configs'/'lab_scan.yaml'


def load_settings(path=CONFIG):
    s = yaml.safe_load(Path(path).read_text())
    s['source'], s['cleaned'] = ROOT/s['source'], ROOT/s['cleaned']
    return s


def floor_transform(meshes, rng=np.random.default_rng(0)):
    """4x4 that levels the floor plane to z = 0 (meshes already Z-up)."""
    V = np.vstack([m.vertices for m in meshes])
    low = V[V[:, 2] < np.percentile(V[:, 2], 40)]
    best = None
    for _ in range(600):
        a, b, c = low[rng.choice(len(low), 3, replace=False)]
        n = np.cross(b-a, c-a)
        if np.linalg.norm(n) < 1e-9:
            continue
        n = n/np.linalg.norm(n)*np.sign(n[2])
        if n[2] < .95:
            continue
        count = int((np.abs((low-a)@n) < .015).sum())
        if best is None or count > best[0]:
            best = (count, n, a)
    _, n, a = best
    floor = V[np.abs((V-a)@n) < .02]
    centre = floor.mean(0)
    n = np.linalg.svd(floor-centre, full_matrices=False)[2][2]
    n *= np.sign(n[2])
    R = trimesh.geometry.align_vectors(n, [0, 0, 1])
    T = np.eye(4); T[:3, 3] = -centre
    back = np.eye(4); back[:2, 3] = centre[:2]      # tilt about the floor's centre; x, y stay the scan's
    return back@R@T


def inside(points, polygon, margin):
    from matplotlib.path import Path as Poly
    return Poly(np.asarray(polygon)).contains_points(points[:, :2], radius=margin) | \
        Poly(np.asarray(polygon)).contains_points(points[:, :2], radius=-margin)


def clean(settings):
    scene = trimesh.load(settings['source'])
    up = trimesh.transformations.rotation_matrix(np.pi/2, [1, 0, 0])      # glTF Y-up -> Z-up
    meshes = []
    for g in scene.geometry.values():
        g = g.copy()
        g.apply_transform(up)
        meshes.append(g)
    level = floor_transform(meshes)
    out, before, after = [], 0, 0
    for g in meshes:
        g.apply_transform(level)
        before += len(g.faces)
        c = g.triangles_center
        keep = inside(c, settings['room_polygon'], settings['margin_m']) & (c[:, 2] < settings['cut_above_m'])
        g.update_faces(keep)
        g.remove_unreferenced_vertices()
        pieces = g.split(only_watertight=False)
        # scan noise: small loose pieces that float (the scan splits real objects into patches too,
        # so only pieces both small and clear of the floor go)
        pieces = [p for p in pieces if len(p.faces) >= settings['min_piece_faces'] or p.area >= settings['min_piece_area_m2']
                  or p.bounds[0, 2] < .05]
        if not pieces:
            continue
        g = trimesh.util.concatenate(pieces)
        v = g.vertices.copy()
        flat = np.abs(v[:, 2]) < settings['floor_snap_m']
        v[flat, 2] = 0.
        g.vertices = v
        after += len(g.faces)
        out.append(g)
    # into OptiTrack coordinates: the take-off spot at the origin, +x along the configured direction
    yaw = np.radians(settings['x_axis_deg'])
    move = np.eye(4); move[:2, 3] = -np.asarray(settings['origin_xy'], float)
    turn = trimesh.transformations.rotation_matrix(-yaw, [0, 0, 1])
    for g in out:
        g.apply_transform(turn@move)
    return out, before, after


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', type=Path, default=CONFIG)
    p.add_argument('--render', type=Path, help='directory for check images')
    args = p.parse_args()
    s = load_settings(args.config)
    meshes, before, after = clean(s)
    scene = trimesh.Scene(meshes)
    s['cleaned'].parent.mkdir(parents=True, exist_ok=True)
    scene.export(s['cleaned'])
    lo, hi = scene.bounds
    print(f'{before} faces -> {after}; room x {lo[0]:.2f}..{hi[0]:.2f}, y {lo[1]:.2f}..{hi[1]:.2f}, '
          f'height {hi[2]:.2f} m -> {s["cleaned"]}')
    if args.render:
        sys.path.insert(0, str(ROOT))
        import pyvista as pv
        from aerowhip.ui.planning.scene import add_lab
        pl = pv.Plotter(off_screen=True, window_size=(1600, 1100))
        pl.set_background('#11161c')
        add_lab(pl, s['cleaned'])
        pl.add_mesh(pv.Box(bounds=(-2, 2, -1.5, 1.5, 0, 2.6)).extract_feature_edges(), color='#3dd6c6', line_width=2)
        pl.add_mesh(pv.Sphere(radius=.08, center=(0, 0, .05)), color='#ff7a59')
        pl.add_mesh(pv.Arrow(start=(0, 0, .05), direction=(1, 0, 0), scale=.8), color='#ff7a59')
        pl.camera_position = [(6.5, -8, 5.5), (0, 0, .8), (0, 0, 1)]
        pl.screenshot(args.render/'lab_perspective.png')
        pl.enable_parallel_projection()
        pl.camera_position = [(0, 0, 30), (0, 0, 0), (0, 1, 0)]
        pl.camera.parallel_scale = 7
        pl.screenshot(args.render/'lab_top.png')
        print('images in', args.render)


if __name__ == '__main__':
    main()
