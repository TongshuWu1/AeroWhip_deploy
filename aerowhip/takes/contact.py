"""Cable contact with the drone (user's rule, 2026-10-28): a take in which the cable touches the drone
is a failure and is not used at all; the cable rising above the clamp without touching is allowed.

The drone is stacked cylinders in its body (tracking) frame: the battery (radius 4 cm) from the cable
attachment (the clamp, the lowest point) up to the arms, the arms and plates from the CAD meshes of the
viewer (aerowhip/viewer/assets/drone: radius 7.6 cm, 4.5 to 3.5 cm below the marker plane), and the
propeller guard ring (11.4 cm) from there to 2 cm above the marker plane. A cable passing under the
propellers is clear; one at the guard or the body is not. The cable is its labelled markers and, between two seen
neighbouring markers (or the attachment and the first marker), the straight chord, sampled every
centimetre; the first `ignore_arc_m` of cable from the clamp is attached there and never counts.

Uses:
  contact (a take fails):          clearance below CONTACT_M (1 cm) in CONTACT_FRAMES (3) frames in a row
  planning (every model):          the simulated cable's clearance at least PLANNING_MARGIN_M (5 cm),
                                   from the first marker on (the hanging first link is 8 cm below the clamp)
  in flight (crazyswarm2):         the same envelope on the raw cloud, landing within 5 cm (its own copy)
"""
import json
from pathlib import Path

import numpy as np

ENVELOPE = ((.04, None, -.045), (.076, -.045, -.035), (.114, -.035, .02))   # (radius, bottom, top) m; None: the attachment
CONTACT_M = .01
CONTACT_FRAMES = 3
PLANNING_MARGIN_M = .05
PLANNING_FROM_ARC_M = .08         # the planner checks the simulated cable from (about) the first marker on
IGNORE_ARC_M = .04
VERSION = 1


def envelope_distance(points_body, attachment_z, envelope=ENVELOPE):
    """Signed distance (m) of points (..., 3) in the drone's body frame from the drone (the union of
    the envelope's cylinders): negative inside."""
    p = np.asarray(points_body, float)
    r = np.hypot(p[..., 0], p[..., 1])
    out = None
    for radius, bottom, top in envelope:
        bottom = attachment_z if bottom is None else bottom
        dr = r-radius
        dz = np.maximum(bottom-p[..., 2], p[..., 2]-top)
        inside = (dr <= 0) & (dz <= 0)
        d = np.where(inside, np.maximum(dr, dz), np.hypot(np.maximum(dr, 0.), np.maximum(dz, 0.)))
        out = d if out is None else np.minimum(out, d)
    return out


def cable_samples(markers, root, arc, *, ignore_arc_m=IGNORE_ARC_M, step=.01):
    """Points along the seen cable of one frame: the markers and the chords between seen neighbours
    (the root counts as the arc-0 neighbour), without the first `ignore_arc_m`. markers (10, 3) with NaN
    where hidden; arc (10,) marker distances along the cable."""
    points = np.vstack([root, markers])
    arcs = np.r_[0., arc]
    seen = np.isfinite(points).all(1)
    out = []
    for i in range(len(points)-1):
        if not (seen[i] and seen[i+1]):
            continue
        n = max(1, int(np.ceil((arcs[i+1]-arcs[i])/step)))
        u = np.linspace(0., 1., n+1)
        a = arcs[i]+u*(arcs[i+1]-arcs[i])
        keep = a >= ignore_arc_m
        out.append(points[i]+u[keep, None]*(points[i+1]-points[i]))
    lonely = [k for k in range(1, len(points)) if seen[k] and not (seen[k-1] or (k+1 < len(points) and seen[k+1])) and arcs[k] >= ignore_arc_m]
    if lonely:
        out.append(points[lonely])
    return np.vstack(out) if out else np.zeros((0, 3))


def clearance(markers, positions, rotations, attachment, arc, *, ignore_arc_m=IGNORE_ARC_M):
    """(F,) smallest signed distance of the seen cable from the drone per frame (NaN without cable data).
    markers (F, 10, 3) world, NaN hidden; positions (F, 3); rotations (F, 3, 3) body to world;
    attachment: tracking origin to the clamp, body frame."""
    attachment = np.asarray(attachment, float)
    out = np.full(len(markers), np.nan)
    for f in range(len(markers)):
        R, p = rotations[f], positions[f]
        pts = cable_samples(markers[f], p+R@attachment, arc, ignore_arc_m=ignore_arc_m)
        if len(pts):
            out[f] = float(envelope_distance((pts-p)@R, attachment[2]).min())
    return out


def nodes_clearance(nodes, positions, rotations, attachment, arc_nodes, *, from_arc_m):
    """(F,) clearance of simulated cable nodes (F, n, 3) (node 0 the attachment), nodes at arc >= from_arc_m."""
    attachment = np.asarray(attachment, float)
    use = np.asarray(arc_nodes) >= from_arc_m
    rel = np.einsum('fni,fij->fnj', np.asarray(nodes)[:, use]-np.asarray(positions)[:, None], np.asarray(rotations))
    return envelope_distance(rel, attachment[2]).min(1)


def contact_events(clear, *, below=CONTACT_M, frames=CONTACT_FRAMES):
    """Indices where the clearance first stays below `below` for `frames` frames in a row."""
    low = np.nan_to_num(clear, nan=np.inf) < below
    run, events = 0, []
    for i, b in enumerate(low):
        run = run+1 if b else 0
        if run == frames:
            events.append(i-frames+1)
    return events


def flight_contact(folder, config, *, refresh=False, write=True):
    """The take's contact verdict from its marker cloud, labelled as in flight (the live labeller),
    cached as <folder>/contact.json: {contact, first_t_s, events, min_clearance_m, frames, version}.
    A flight without a cloud (no cable) has no contact."""
    folder = Path(folder)
    cache = folder/'contact.json'
    if cache.exists() and not refresh:
        try:
            data = json.loads(cache.read_text())
            if data.get('version') == VERSION and cache.stat().st_mtime >= (folder/'poses.jsonl').stat().st_mtime:
                return data
        except (OSError, ValueError):
            pass
    params = json.loads((folder/'parameters.json').read_text())
    setup = str((params.get('settings') or {}).get('setup', ''))
    if not (folder/'cloud.jsonl').exists() or setup.endswith('_nocable'):
        data = dict(contact=False, reason='no cable', version=VERSION)
    else:
        from aerowhip.online.stream import FlightStream
        from aerowhip.physics.state import quaternion_xyzw_to_rotation_matrix
        arc = np.cumsum(config['cable']['marker_interval_lengths_m'])
        attachment = config['attachment']['position_tracking_m']
        stream = FlightStream(folder, arc, attachment, keep_s=1e9, replay=True)
        stream.advance(np.inf)
        poses = {round(p['stamp'], 6): p for p in stream.poses}
        stamps, markers, positions, rotations = [], [], [], []
        for stamp, m in stream.labels:
            pose = poses.get(round(stamp, 6))
            if pose is None:
                continue
            stamps.append(stamp); markers.append(m)
            positions.append(pose['position']); rotations.append(quaternion_xyzw_to_rotation_matrix(np.asarray(pose['quaternion_xyzw'], float)))
        clear = clearance(np.array(markers), np.array(positions, float), np.array(rotations), attachment, arc) if markers else np.zeros(0)
        events = contact_events(clear)
        t0 = stamps[0] if stamps else 0.
        data = dict(contact=bool(events), first_t_s=round(stamps[events[0]]-t0, 2) if events else None,
                    events=[round(stamps[i]-t0, 2) for i in events][:20],
                    min_clearance_m=round(float(np.nanmin(clear)), 4) if np.isfinite(clear).any() else None,
                    frames=len(clear), version=VERSION)
    try:
        if write:
            cache.write_text(json.dumps(data)+'\n')
    except OSError:
        pass                                        # a read-only folder: the verdict is still returned
    return data
