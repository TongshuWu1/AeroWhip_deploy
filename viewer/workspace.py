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

