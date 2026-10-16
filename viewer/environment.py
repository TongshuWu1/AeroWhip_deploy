"""Task-independent laboratory floor and lighting, for display only."""

import numpy as np

from simulation.configuration import ViewerConfiguration


def add_environment(plotter, pv, viewer: ViewerConfiguration) -> None:
    half = viewer.ground_size_m / 2
    z = viewer.ground_z_m
    floor = pv.Plane(center=(0, 0, z), direction=(0, 0, 1),
                     i_size=viewer.ground_size_m * 5, j_size=viewer.ground_size_m * 5,
                     i_resolution=1, j_resolution=1)
    plotter.add_mesh(floor, name="environment/floor", color="#99a8b5",
                     pbr=True, metallic=0.0, roughness=0.95, pickable=False)

    # Thin filled strips give grid lines a consistent metric width. A small
    # offset avoids coplanar depth flicker; they are visual floor markings.
    spacing = viewer.grid_spacing_m
    indices = range(-int(np.floor(half / spacing)), int(np.floor(half / spacing)) + 1)
    for major in (False, True):
        points, faces = [], []
        width = 0.005 if major else 0.003
        for index in indices:
            value = index * spacing
            if bool(np.isclose(value, round(value), atol=1e-9)) != major:
                continue
            for along_x in (False, True):
                strip = [(-half, value - width / 2), (half, value - width / 2),
                         (half, value + width / 2), (-half, value + width / 2)]
                if not along_x:
                    strip = [(y, x) for x, y in reversed(strip)]
                start = len(points)
                points.extend((x, y, z + 0.00015) for x, y in strip)
                faces.extend((4, start, start + 1, start + 2, start + 3))
        if points:
            plotter.add_mesh(pv.PolyData(np.asarray(points), faces),
                             name=f"environment/grid-{'major' if major else 'minor'}",
                             color="#526477" if major else "#74879a",
                             pbr=True, metallic=0.0, roughness=0.95, pickable=False)


def configure_lighting(plotter, pv, viewer: ViewerConfiguration) -> None:
    background = viewer.background_rgb
    plotter.set_background(background, top=tuple(min(1.0, value + 0.12)
                                                 for value in background))
    plotter.remove_all_lights()
    for position, intensity, color in (
        ((-3.0, -4.0, 7.0), 1.5, "#fff5e6"),
        ((4.0, -1.0, 4.0), 0.7, "#e3efff"),
        ((0.0, 4.0, 6.0), 0.7, "#ffffff"),
    ):
        plotter.add_light(pv.Light(position=position, focal_point=(0, 0, 1),
                                   intensity=intensity, color=color,
                                   positional=False, light_type="scene light"))
    plotter.add_light(pv.Light(light_type="headlight", intensity=1.3))
