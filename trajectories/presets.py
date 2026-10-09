"""Small analytic motion references; no controller, UI or physics dependencies."""

import numpy as np

from trajectories.full_state import COLUMNS, FullStateTrajectory


MOTIONS = ('Hover', 'Move', 'Back and forth')


def simple_motion(kind, position, yaw=0., *, displacement=(.25, 0., 0.),
                  duration_s=4., timestep_s=1/30):
    """Rest-to-rest quintic motion, fixed yaw; duration covers the whole motion.

    Back and forth spends half the duration on each leg. Packets use the nominal
    command grid plus exact end/turnaround timestamps when off-grid. Afterward
    FullStateTrajectory holds the final position; no simulated state is imposed.
    """
    position, displacement = np.asarray(position, dtype=float), np.asarray(displacement, dtype=float)
    if (kind not in MOTIONS or position.shape != (3,) or displacement.shape != (3,)
            or not np.isfinite(position).all() or not np.isfinite(displacement).all()
            or not np.isfinite([yaw, duration_s, timestep_s]).all()
            or duration_s <= 0 or timestep_s <= 0):
        raise ValueError('Choose a motion with finite XYZ/yaw and positive duration/timestep.')
    count = int(np.ceil(duration_s/timestep_s))
    if count > 100_000:
        raise ValueError('Motion exceeds 100,000 command intervals.')
    extra = [duration_s/2, duration_s] if kind == 'Back and forth' else [duration_s]
    time = np.unique(np.r_[np.arange(count)*timestep_s, extra])
    # Avoid nearly coincident boundary/grid timestamps caused by roundoff.
    from simulation.timing import TIME_EPS
    time = time[np.r_[np.diff(time) > TIME_EPS, True]]
    time[0] = 0.
    rows = np.zeros((len(time), len(COLUMNS)))
    jerk = np.zeros((len(time),3))
    rows[:, 0], rows[:, 1:4], rows[:, 10] = time, position, yaw
    if kind != 'Hover':
        leg = duration_s/(2 if kind == 'Back and forth' else 1)
        returning = (time > leg) if kind == 'Back and forth' else np.zeros(len(time), dtype=bool)
        s = np.clip(np.where(returning, time-leg, time)/leg, 0., 1.)
        shape = 10*s**3-15*s**4+6*s**5
        velocity = (30*s**2-60*s**3+30*s**4)/leg
        acceleration = (60*s-180*s**2+120*s**3)/leg**2
        shape = np.where(returning, 1-shape, shape)
        sign = np.where(returning, -1., 1.)
        rows[:, 1:4] += shape[:, None]*displacement
        rows[:, 4:7] = (sign*velocity)[:, None]*displacement
        rows[:, 7:10] = (sign*acceleration)[:, None]*displacement
        # Right-hand derivative at the C2 turnaround. PVA is unchanged; a
        # quintic leg has a finite jerk jump, not a C3/smooth-rate guarantee.
        jr = (time >= leg) if kind == 'Back and forth' else np.zeros(len(time),dtype=bool)
        js = np.clip(np.where(jr,time-leg,time)/leg,0.,1.)
        jerk[:] = ((np.where(jr,-1.,1.)*(60-360*js+360*js**2)/leg**3)[:,None]*displacement)
        jerk[-1] = 0.  # The terminal packet begins the stationary hold.
    return FullStateTrajectory(rows, jerk_world=jerk)
