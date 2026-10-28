"""Windows cut from a recorded flight at a phase (a part of a mixed or planned flight), whether or
not the cable labels survive inside them: only the history before the window must be fully
labelled (the replay's initial cable state); later frames may have hidden markers."""
import numpy as np

from aerowhip.takes.calibration_flights import DT, Window, quaternion_xyzw_to_rotation_matrix


def phase_window(flight, phase, seconds, history_s=.1, lead_s=0.):
    """The window of `seconds` from `lead_s` before the first frame of `phase` (exact phase name),
    its history the fully labelled frames just before. None when the phase or the labels are missing."""
    stamp = flight.stamp
    phases = [flight.phase_at(s) for s in stamp]
    try:
        first = next(i for i, p in enumerate(phases) if p == phase)
    except StopIteration:
        return None
    first = max(first-int(round(lead_s/DT)), 0)
    hist, steps = int(round(history_s/DT)), int(round(seconds/DT))
    if flight.markers is None:
        return None
    while first > hist and not np.isfinite(flight.markers[first-hist:first+1]).all():
        first -= 1
    if first < hist or first+steps+1 > flight.frames or not np.isfinite(flight.markers[first-hist:first+1]).all():
        return None
    sel = slice(first-hist, first+steps+1)
    m = flight.markers[sel]
    seen = np.isfinite(m).all(2)
    R = np.stack([quaternion_xyzw_to_rotation_matrix(q) for q in flight.quaternion[sel]])
    t = (flight.frame[sel]-flight.frame[first])*DT
    return Window(flight.folder.name, phase, float(stamp[first]), np.arange(steps+1)*DT,
                  flight.position[sel][hist:].copy(), R[hist:].copy(), m[hist:].copy(), seen[hist:].copy(),
                  flight.command_at(stamp[sel][hist:]), flight.command_at(stamp[sel][hist:], strip_trim=False),
                  flight.position[sel][:hist+1].copy(), R[:hist+1].copy(), m[:hist+1].copy(), t[:hist+1].copy())


def raw_cloud_rise(folder, window, attachment_tracking):
    """Per window frame, the highest raw-cloud point (within 1.1 m of the drone, not its own markers)
    above the attachment, m, from cloud.jsonl (NaN where no cable point is seen). The labels can
    vanish while the cable is over the drone; the raw cloud still shows it."""
    import json
    clouds = [json.loads(l) for l in (folder/'cloud.jsonl').read_text().splitlines() if l.strip()]
    stamps = np.array([c['stamp'] for c in clouds])
    out = np.full(len(window.time)-1, np.nan)
    for k in range(len(out)):
        s = window.start_stamp+window.time[k+1]
        i = min(int(np.searchsorted(stamps, s)), len(stamps)-1)
        pts = np.array([q[:3] for q in clouds[i]['points']], float)
        if not len(pts):
            continue
        drone = window.position[k+1]
        root_z = (drone+window.rotation[k+1]@np.asarray(attachment_tracking))[2]
        rel = pts-drone
        dist = np.linalg.norm(rel, axis=1)
        ok = (dist > .12) & (dist < 1.1)
        if ok.any():
            out[k] = pts[ok, 2].max()-root_z
    return out
