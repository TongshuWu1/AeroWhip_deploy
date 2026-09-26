"""Command events and sampled feedback, independent of the integration step."""

from dataclasses import dataclass
import math

import numpy as np


TIME_EPS = 1e-12  # Timestamp roundoff only, not a command-period quantizer.


def integration_steps(duration_s, maximum_step_s):
    """Subdivide a held-command interval without enlarging the physics step."""
    if not np.isfinite([duration_s, maximum_step_s]).all() or min(duration_s, maximum_step_s) <= 0:
        raise ValueError('Integration duration and maximum step must be positive and finite')
    ratio = duration_s / maximum_step_s
    nearest = round(ratio)
    if nearest >= 1 and abs(duration_s-nearest*maximum_step_s) <= TIME_EPS:
        return nearest, maximum_step_s
    count = math.ceil(ratio)
    return count, duration_s/count


def validate_command_times(times):
    times = np.array(times, dtype=np.float64, copy=True)
    if (times.ndim != 1 or not len(times) or not np.isfinite(times).all()
            or times[0] != 0 or np.any(np.diff(times) <= TIME_EPS)):
        raise ValueError('Command times must start at zero and increase strictly')
    return times


@dataclass(frozen=True)
class CommandSchedule:
    """Each part is (duration, command row); -1 denotes the initial hold.

    Commands are right-continuous zero-order holds. A command at an output's
    right boundary acts in the next output interval, never the preceding one.
    """
    time_s: np.ndarray
    requested_indices: np.ndarray
    applied_indices: np.ndarray
    parts: tuple

    @classmethod
    def build(cls, command_times, duration_s, feedback_timestep_s, delay_s):
        times = validate_command_times(command_times)
        if (not np.isfinite([duration_s, feedback_timestep_s, delay_s]).all()
                or min(duration_s, feedback_timestep_s) <= 0 or delay_s < 0):
            raise ValueError('Positive duration/feedback step and nonnegative delay are required')
        count = round(duration_s/feedback_timestep_s)
        if count < 1 or abs(count*feedback_timestep_s-duration_s) > TIME_EPS:
            raise ValueError('Duration must be a whole number of feedback intervals')
        output = np.arange(count+1)*feedback_timestep_s
        delayed = times+delay_s
        requested = np.searchsorted(times, output[:-1]+TIME_EPS, side='right')-1
        applied = np.searchsorted(delayed, output[:-1]+TIME_EPS, side='right')-1
        parts = []
        for start, end, index in zip(output[:-1], output[1:], applied):
            cursor, row, segments = start, int(index), []
            while row+1 < len(delayed) and delayed[row+1] < end-TIME_EPS:
                event = delayed[row+1]
                if event > cursor+TIME_EPS:
                    segments.append((float(event-cursor), row))
                cursor, row = max(cursor, event), row+1
            segments.append((float(end-cursor), row))
            parts.append(tuple(segments))
        return cls(output, requested, applied, tuple(parts))
