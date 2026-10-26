"""Explicit kinematic cable controls for measurement checks, without physics.

Synthetic arrays are geometric constructions, not dynamically feasible flights
or labels of whipping.
"""

import numpy as np


def _chain(rest, angles, rates):
    """Integrate planar unit tangents; velocities are analytic derivatives."""
    edges = rest[None, :, None]*np.stack((np.sin(angles), np.zeros_like(angles),
                                        -np.cos(angles)), axis=-1)
    changes = rest[None, :, None]*rates[..., None]*np.stack(
        (np.cos(angles), np.zeros_like(angles), np.sin(angles)), axis=-1)
    zero = np.zeros((len(angles), 1, 3))
    return (np.concatenate((zero, np.cumsum(edges, axis=1)), axis=1),
            np.concatenate((zero, np.cumsum(changes, axis=1)), axis=1))


def kinematic_controls(rest_lengths, *, time_s=None, origin=(0., 0., 2.),
                       attachment_offset=(0., 0., -.051)):
    """Return deterministic, constant-edge-length controls and construction notes.

    Time and length scales below define fixtures, never classification thresholds.
    """
    rest = np.asarray(rest_lengths, dtype=float)
    t = np.arange(451)*.01 if time_s is None else np.asarray(time_s, dtype=float)
    if (rest.ndim != 1 or len(rest) < 2 or not np.isfinite(rest).all()
            or np.any(rest <= 0) or t.ndim != 1 or len(t) < 2
            or not np.isfinite(t).all() or np.any(np.diff(t) <= 0)):
        raise ValueError('Positive edge lengths and increasing finite times required')
    origin, offset = np.asarray(origin, float), np.asarray(attachment_offset, float)
    if (origin.shape != (3,) or offset.shape != (3,)
            or not np.isfinite(origin).all() or not np.isfinite(offset).all()):
        raise ValueError('Finite XYZ origin and attachment offset required')
    length = rest.sum()
    material = np.cumsum(rest)[:-1]
    time = t-t[0]
    duration = time[-1]

    def wrap(p, v, shift=None, shift_rate=None):
        shift = np.zeros((len(t), 3)) if shift is None else shift
        shift_rate = np.zeros_like(shift) if shift_rate is None else shift_rate
        return dict(time_s=t.copy(), cable_positions=p+origin+offset+shift[:, None],
                    cable_velocities=v+shift_rate[:, None],
                    drone_positions=origin+shift, drone_velocities=shift_rate.copy(),
                    drone_rotations=np.tile(np.eye(3), (len(t), 1, 1)),
                    drone_angular_velocities=np.zeros_like(shift),
                    state_valid=np.ones(len(t), bool),
                    cable_marker_indices=np.arange(len(rest)+1),
                    evidence_kind=np.asarray('synthetic_kinematic_control'))

    def wave(amplitude, backward=False, horizontal=False):
        # Prescribe signed turns at interior material points, yielding one
        # positive curvature packet instead of identifying a moving centroid.
        speed = 1.8*length/duration
        center = (-.25*length+speed*time if not backward else
                  1.25*length-speed*time)
        rate = -speed if backward else speed
        width = .12*length
        turns = amplitude*np.exp(-.5*((material[None]-center[:, None])/width)**2)
        turn_rate = turns*(material[None]-center[:, None])*rate/width**2
        angles = np.column_stack((np.zeros(len(t)), np.cumsum(turns, axis=1)))
        rates = np.column_stack((np.zeros(len(t)), np.cumsum(turn_rate, axis=1)))
        if horizontal:
            angles += np.pi/2
        return _chain(rest, angles, rates)

    base = np.tile(np.linspace(0., .9, len(rest)), (len(t), 1))
    p, _ = _chain(rest, base, np.zeros_like(base))
    theta = 2.*time
    cosine, sine = np.cos(theta), np.sin(theta)
    rotation = np.zeros((len(t), 3, 3))
    rotation[:, 0, 0] = rotation[:, 2, 2] = cosine
    rotation[:, 0, 2], rotation[:, 2, 0], rotation[:, 1, 1] = sine, -sine, 1.
    rotated = np.einsum('tij,tnj->tni', rotation, p)
    velocity = np.cross(np.array([0., 2., 0.]), rotated)
    result = [dict(name='rigid_curved_rotation', arrays=wrap(rotated, velocity),
                   notes='Fixed curved material shape rotates about the attachment at 2 rad/s; analytic velocities. No shape change, though the tip moves quickly.')]
    shift = np.column_stack((.8*time, np.zeros_like(time), np.zeros_like(time)))
    result.append(dict(name='translated_curved_shape', arrays=wrap(
        p, np.zeros_like(p), shift, np.tile([.8, 0., 0.], (len(t), 1))),
        notes='Unchanging curved cable translated uniformly at 0.8 m/s along world X.'))
    for name, backwards in [('small_travelling_wave', False), ('small_backward_wave', True)]:
        pos, vel = wave(.01, backward=backwards)
        result.append(dict(name=name, arrays=wrap(pos, vel), notes=(
            'Prescribed 0.01-rad Gaussian interior-turn packet moves '+
            ('tip toward attachment' if backwards else 'attachment toward tip')+
            ' in material coordinates; small amplitude is specified, not a whip label.')))
    fraction = time/duration
    turns = np.zeros((len(t), len(rest)-1))
    rates = np.zeros_like(turns)
    left, right = max(0, len(rest)//4-1), min(len(rest)-2, 3*len(rest)//4)
    turns[:, left], turns[:, right] = .25*(1-fraction), .25*fraction
    rates[:, left], rates[:, right] = -.25/duration, .25/duration
    angles = np.column_stack((np.zeros(len(t)), np.cumsum(turns, axis=1)))
    angular_rate = np.column_stack((np.zeros(len(t)), np.cumsum(rates, axis=1)))
    pos, vel = _chain(rest, angles, angular_rate)
    result.append(dict(name='separated_bends_fade_and_grow', arrays=wrap(pos, vel),
        notes='Two bends stay at distinct material nodes while one fades and the other grows; their curvature centroid shifts without a bend travelling between them.'))
    # A compactly supported time envelope makes the later segment exactly rigid.
    phase = np.clip(time/(.4*duration), 0., 1.)
    envelope = np.sin(np.pi*phase)**4
    envelope_rate = (4*np.pi/(.4*duration)*np.sin(np.pi*phase)**3*np.cos(np.pi*phase))
    envelope[phase >= 1.] = envelope_rate[phase >= 1.] = 0.
    center = -.2*length+3.5*length*time/duration
    width = .12*length
    gaussian = .3*np.exp(-.5*((material[None]-center[:, None])/width)**2)
    turns = gaussian*envelope[:, None]
    rates = gaussian*(envelope_rate[:, None]+envelope[:, None]*
                      (material[None]-center[:, None])*(3.5*length/duration)/width**2)
    angles = np.pi/2+np.column_stack((np.zeros(len(t)), np.cumsum(turns, axis=1)))
    angular_rate = np.column_stack((np.zeros(len(t)), np.cumsum(rates, axis=1)))
    pos, vel = _chain(rest, angles, angular_rate)
    phase = np.clip((time/duration-.72)/.13, 0., 1.)
    shift = np.zeros((len(t), 3)); shift_rate = np.zeros_like(shift)
    shift[:, 0] = .6*(3*phase**2-2*phase**3)
    shift_rate[:, 0] = .6*(6*phase-6*phase**2)/(.13*duration)
    arrays = wrap(pos, vel, shift, shift_rate)
    index = lambda f: int(np.argmin(abs(time-f*duration)))
    result.append(dict(name='earlier_wave_later_rigid_crossing', arrays=arrays,
        events=[dict(start_index=0, peak_index=index(.22), end_index=index(.4)),
                dict(start_index=index(.65), peak_index=index(.79), end_index=index(.95))],
        notes='Prescribed early travelling turn packet becomes exactly zero by 40% of duration. A later rigid horizontal translation crosses the frozen +X target plane. Explicit windows are construction intervals, not physical labels.'))
    return result
