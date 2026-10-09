"""Visual workspace guides, independent of task constraints and physics."""

import numpy as np


def add_cylinder_boundary(plotter, *, center_xy_world_m, radius_m,
                          height_lower_world_m, height_upper_world_m):
    """Draw two rings and four uprights; return the guide's world bounds."""
    import pyvista as pv
    count = 64
    angle = np.arange(count)*2*np.pi/count
    xy = np.asarray(center_xy_world_m)+radius_m*np.column_stack((np.cos(angle), np.sin(angle)))
    points = np.concatenate([np.column_stack((xy, np.full(count, z)))
                             for z in (height_lower_world_m, height_upper_world_m)])
    index = np.arange(count)
    segments = np.concatenate((np.column_stack((index, (index+1) % count)),
                               np.column_stack((count+index, count+(index+1) % count)),
                               np.column_stack((index[::16], count+index[::16]))))
    lines = np.column_stack((np.full(len(segments), 2), segments)).ravel()
    plotter.add_mesh(pv.PolyData(points, lines=lines), name='workspace', label='Workspace',
                     color='#805ac0', line_width=2, opacity=.7, lighting=False, pickable=False)
    return np.column_stack((points.min(0), points.max(0))).ravel()


def add_strike_surface(plotter, *, center_world_m, normal_world, width_m, height_m):
    """Draw a virtual vertical strike region and entry arrow; no contact forces.

    The horizontal unit normal points in the requested entry direction.
    Return the rectangle's four world-space corners for camera framing.
    """
    import pyvista as pv
    center, normal = np.asarray(center_world_m), np.asarray(normal_world)
    if (center.shape != (3,) or normal.shape != (3,)
            or not np.isfinite(center).all() or not np.isfinite(normal).all()
            or not np.isclose(np.linalg.norm(normal), 1., rtol=0, atol=1e-12)
            or abs(normal[2]) > 1e-12
            or not np.isfinite([width_m, height_m]).all() or min(width_m, height_m) <= 0):
        raise ValueError('Provide a finite center, horizontal unit normal and positive region size')
    side = np.array([-normal[1], normal[0], 0.])*width_m/2
    up = np.array([0., 0., height_m/2])
    points = np.array([center-side-up, center+side-up, center+side+up, center-side+up])
    plotter.add_mesh(pv.PolyData(points, faces=[4, 0, 1, 2, 3]),
        name='strike_region', color='#397dcc', opacity=.15, lighting=False, pickable=False)
    plotter.add_mesh(pv.lines_from_points(np.vstack((points, points[:1]))),
        name='strike_outline', label='Strike region', color='#397dcc', line_width=3,
        lighting=False, pickable=False)
    plotter.add_mesh(pv.Arrow(start=center-.25*normal, direction=normal, scale=.5),
        name='strike_direction', label='Entry direction', color='#397dcc', lighting=False, pickable=False)
    return points


def add_release_height(plotter, *, lower_xy_world_m, upper_xy_world_m, height_world_m):
    """Draw a requested cable-tip height; return four corners for camera framing."""
    import pyvista as pv
    lower, upper = np.asarray(lower_xy_world_m), np.asarray(upper_xy_world_m)
    if (lower.shape != (2,) or upper.shape != (2,)
            or not np.isfinite(np.r_[lower, upper, height_world_m]).all()
            or not (lower < upper).all()):
        raise ValueError('Provide finite ordered XY bounds and a finite height')
    points = np.array([[x, y, height_world_m] for x, y in
        ((lower[0], lower[1]), (upper[0], lower[1]), (upper[0], upper[1]), (lower[0], upper[1]))])
    plotter.add_mesh(pv.PolyData(points, faces=[4, 0, 1, 2, 3]), name='release_height_plane',
        color='#b8942a', opacity=.10, lighting=False, pickable=False)
    plotter.add_mesh(pv.lines_from_points(np.vstack((points, points[:1]))),
        name='release_height_outline', label='Desired height', color='#b8942a',
        opacity=.7, line_width=2, lighting=False, pickable=False)
    return points


def add_event_marker(plotter, position_world_m):
    """Mark a caller-selected event; the viewer does not detect or score events."""
    import pyvista as pv
    position = np.asarray(position_world_m)
    if position.shape != (3,) or not np.isfinite(position).all():
        raise ValueError('Event position must contain three finite world coordinates')
    plotter.add_mesh(pv.Sphere(radius=.017, center=position), name='release_point',
        label='Scored event', color='#df3c4b', lighting=False, pickable=False)
    for axis in range(3):
        arm = np.eye(3)[axis]*.04
        plotter.add_mesh(pv.Line(position-arm, position+arm), name=f'release_cross_{axis}',
            color='#df3c4b', line_width=3, lighting=False, pickable=False)
