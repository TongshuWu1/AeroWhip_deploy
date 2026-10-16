"""Measured CAD frame with visual-only accessories estimated from the drone video."""

from pathlib import Path
from dataclasses import dataclass

import numpy as np

from simulation.configuration import ViewerConfiguration
from simulation.state import SimulationState


ASSET_DIRECTORY = Path(__file__).resolve().parent / "assets" / "drone"


@dataclass(frozen=True)
class RotorVisual:
    actors: tuple
    center_body_m: tuple[float, float, float]
    direction: float


def load_frame_meshes(pv, attachment_body_m, viewer: ViewerConfiguration):
    """Return lower body and upper guard in the existing drone reference frame.

    The supplied STLs are interpreted as millimeters. Rotate +90 degrees about
    X to map CAD +Y to simulation +Z without reflecting the geometry. The frame
    stays centered in XY; the cable attachment comes from the configured state.
    """
    body = pv.read(ASSET_DIRECTORY / "body_v2.stl").clean()
    guard = pv.read(ASSET_DIRECTORY / "guard_v5.stl").clean()
    for mesh in (body, guard):
        mesh.scale(0.001, inplace=True)
        mesh.rotate_x(90.0, inplace=True)

    body_bottom = attachment_body_m[2] + viewer.cable_attachment_drop_m
    # The central plate underside is CAD Y=0 (now Z=0). Motor bosses extend
    # down to CAD Y=-5 mm, so the whole-mesh minimum is not the contact face.
    body.translate((0.0, 0.0, body_bottom), inplace=True)
    guard_bottom = body.bounds.z_max + viewer.plate_gap_m
    guard.translate((0.0, 0.0, guard_bottom - guard.bounds.z_min), inplace=True)
    return body, guard


def _propeller_blade(pv, radius: float, handedness: float):
    """A tapered, swept, pitched blade; appearance only, not an aero model."""
    stations = np.linspace(0.10, 1.0, 22)
    angles = np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False)
    points = []
    for fraction in stations:
        width = radius * (0.045 + 0.16 * np.sin(np.pi * fraction) ** 0.8)
        if fraction == 1.0:
            width = radius * 0.012
        pitch = handedness * np.deg2rad(30.0 - 15.0 * fraction)
        sweep = handedness * radius * 0.12 * fraction**1.5
        for angle in angles:
            chord = width * np.cos(angle)
            thickness = 0.00055 * np.sin(angle)
            points.append((
                radius * fraction,
                sweep + chord * np.cos(pitch) - thickness * np.sin(pitch),
                chord * np.sin(pitch) + thickness * np.cos(pitch),
            ))
    faces = []
    for ring in range(len(stations) - 1):
        for vertex in range(len(angles)):
            a = ring * len(angles) + vertex
            b = ring * len(angles) + (vertex + 1) % len(angles)
            faces.extend((4, a, b, b + len(angles), a + len(angles)))
    faces.extend((len(angles), *reversed(range(len(angles)))))
    faces.extend((len(angles), *range(len(points) - len(angles), len(points))))
    return pv.PolyData(np.asarray(points), faces).triangulate()


def add_drone_model(plotter, pv, state: SimulationState, viewer: ViewerConfiguration, *, name='drone'):
    rotation = state.drone.rotation_body_to_world
    attachment = rotation.T @ (
        state.attachment_position_world_m - state.drone.position_world_m
    )
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = state.drone.position_world_m
    actors = []
    rotors = []
    prefix = f'robot/{name}'

    def add(mesh, name, color, metallic=0.0, roughness=0.5, opacity=1.0):
        actor = plotter.add_mesh(
            mesh.transform(transform, inplace=False), name=f"{prefix}/{name}",
            color=color, smooth_shading=True, split_sharp_edges=True,
            pbr=True, metallic=metallic, roughness=roughness,
            opacity=opacity, specular=0.5, specular_power=35,
        )
        actors.append(actor)
        return actor

    def cylinder(center, radius, height):
        return pv.Cylinder(center=center, direction=(0, 0, 1), radius=radius,
                           height=height, resolution=48)

    def box(center, dimensions):
        return pv.Cube(center=center, x_length=dimensions[0],
                       y_length=dimensions[1], z_length=dimensions[2])

    def wire(points, name, color="#202126", radius=0.00065):
        add(pv.Spline(np.asarray(points), 40).tube(radius=radius, n_sides=10),
            name, color, roughness=0.6)

    body, guard = load_frame_meshes(pv, attachment, viewer)
    bottom = attachment[2] + viewer.cable_attachment_drop_m
    top = body.bounds.z_max
    guard_bottom, guard_top = guard.bounds.z_min, guard.bounds.z_max
    yellow = "#ffc900"
    add(body, "body", yellow, roughness=0.65)
    add(guard, "guard", yellow, roughness=0.65)

    # Accessories below are estimates from the video, not calibrated geometry.
    for index, (x, y) in enumerate(((0.015, 0.015), (-0.015, 0.015),
                                    (-0.015, -0.015), (0.015, -0.015))):
        add(cylinder((x, y, (top + guard_bottom) / 2), 0.002,
                     viewer.plate_gap_m), f"spacer-{index}", "#20232b")
        add(cylinder((x, y, guard_top + 0.0005), 0.0024, 0.001),
            f"screw-{index}", "#bac0c5", metallic=0.7, roughness=0.25)

    for index, z in enumerate((top + 0.006, guard_bottom - 0.004)):
        add(box((0, 0, z), (0.034, 0.034, 0.0016)),
            f"pcb-{index}", "#24372f", roughness=0.65)
        add(box((0, 0, z + 0.002), (0.012, 0.013, 0.0025)),
            f"chip-{index}", "#14171d")
        for side in (-1, 1):
            add(box((side * 0.009, 0.014, z + 0.003), (0.007, 0.005, 0.005)),
                f"socket-{index}-{side}", "#dddccf")
    label_z = guard_bottom - 0.001
    add(cylinder((0, 0, label_z), 0.010, 0.0006), "id-disc", "#f4f0df")
    identity = pv.Text3D(name, depth=0.0001, height=0.006)
    identity.scale(min(1., .017 / max(identity.bounds.x_max-identity.bounds.x_min, 1e-9)), inplace=True)
    identity.translate(np.array((0, 0, label_z + 0.0004))-identity.center, inplace=True)
    add(identity, "id-number", "#25272a")

    # CAD motor centers are at (+/-50, +/-50) mm in the plate plane.
    for index, (x, y) in enumerate(((0.05, 0.05), (-0.05, 0.05),
                                    (-0.05, -0.05), (0.05, -0.05))):
        add(cylinder((x, y, top + 0.003), 0.008, 0.006),
            f"motor-base-{index}", "#453466", metallic=0.55)
        rotating = [add(cylinder((x, y, top + 0.009), 0.0085, 0.006),
            f"motor-bell-{index}", "#daa329", metallic=0.65, roughness=0.28)]
        prop_z = top + 0.015
        rotating.append(add(cylinder((x, y, prop_z), 0.0045, 0.003),
            f"hub-{index}", "#771328", metallic=0.18, roughness=0.25))
        for blade_index in range(3):
            blade = _propeller_blade(pv, viewer.rotor_radius_m,
                                      1.0 if index % 2 == 0 else -1.0)
            blade.rotate_z(120 * blade_index + 20 * index, inplace=True)
            blade.translate((x, y, prop_z), inplace=True)
            rotating.append(add(blade, f"blade-{index}-{blade_index}", "#a90e27",
                roughness=0.24))
        rotating.append(add(cylinder((x, y, prop_z + 0.002), 0.0018, 0.001),
            f"hub-screw-{index}", "#c9ccd0", metallic=0.8))
        rotors.append(RotorVisual(tuple(rotating), (x, y, prop_z),
                                  1.0 if index % 2 == 0 else -1.0))
        add(pv.Cone(center=(x, y, bottom - 0.011), direction=(0, 0, -1),
                    height=0.022, radius=0.006, resolution=40),
            f"leg-{index}", yellow, roughness=0.65)
        for strand in range(3):
            d = strand * 0.0014
            wire([(x, y, top + 0.001), (x * 0.75, y * 0.8 + d, bottom - 0.005),
                  (x * 0.4, y * 0.45 + d, bottom - 0.008),
                  (x * 0.2, y * 0.2 + d, top + 0.005)],
                 f"motor-wire-{index}-{strand}", radius=0.00045)

    # The measured 15 mm attachment drop is the battery thickness: its top
    # touches the lower plate and the cable root lies on its bottom face.
    # Length and width remain visual estimates; root XY comes from the state.
    battery_height = viewer.cable_attachment_drop_m
    battery_width = 0.030
    battery_z = bottom - battery_height / 2
    battery_bottom = bottom - battery_height
    add(box((0, 0, battery_z), (0.062, battery_width, battery_height)),
        "battery", "#c8c8cf", metallic=0.45, roughness=0.35)
    add(box((0.029, 0, battery_z), (0.004, battery_width, battery_height)),
        "battery-end", "#76558d", metallic=0.2)

    # A narrow zip tie loops over the lower frame and around the battery,
    # crossing the cable root at the underside. No separate suspension stem.
    tie_x = attachment[0]
    tie_width, tie_thickness = 0.0025, 0.0005
    for side in (-1, 1):
        add(box((tie_x, side * (battery_width + tie_thickness) / 2,
                 (top + battery_bottom) / 2),
                (tie_width, tie_thickness, top - battery_bottom)),
            f"zip-tie-side-{side}", "#202227", roughness=0.8)
    for name, z in (("top", top + tie_thickness / 2),
                    ("bottom", battery_bottom + tie_thickness / 2)):
        add(box((tie_x, 0, z),
                (tie_width, battery_width + 2 * tie_thickness, tie_thickness)),
            f"zip-tie-{name}", "#202227", roughness=0.8)
    add(box((tie_x, -battery_width / 2 - 0.001, battery_bottom + 0.002),
            (0.004, 0.0025, 0.0035)),
        "zip-tie-lock", "#202227", roughness=0.8)

    for i, color in enumerate(("#ad2035", "#1e2026")):
        wire([(0.03, i * 0.003, battery_z),
              (0.045, 0.018 + i * 0.003, battery_z - 0.002),
              (0.047, -0.007 + i * 0.003, bottom - 0.012)],
             f"battery-lead-{i}", color, 0.0012)
    add(box((0.047, -0.010, bottom - 0.012), (0.008, 0.009, 0.006)),
        "power-connector", "#dab227")

    for index, (x, y) in enumerate(((0.054, 0), (0, 0.054),
                                    (-0.054, 0), (0, -0.074))):
        add(cylinder((x, y, guard_top + 0.0025), 0.0035, 0.005),
            f"marker-stem-{index}", "#eee9d8")
        add(pv.Sphere(center=(x, y, guard_top + 0.010), radius=0.006,
                      theta_resolution=32, phi_resolution=24),
            f"tracking-marker-{index}", "#9c9d9c", roughness=0.7)

    spinning_ids = {id(actor) for rotor in rotors for actor in rotor.actors}
    return [actor for actor in actors if id(actor) not in spinning_ids], rotors
