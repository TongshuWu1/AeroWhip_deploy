"""Short measured-state segments; complete flights remain separate for replay."""

from dataclasses import replace
import numpy as np

from aerowhip.fitting.coupled_fit import window_weights
from aerowhip.fitting.prepare_window import initialize


def prepare_residual_segments(model, windows, metadata, *, duration_s, history_s):
    if not np.isfinite([duration_s, history_s]).all():
        raise ValueError('Finite segment and history durations required')
    steps = round(duration_s/model.dt)
    history = round(history_s/model.dt)
    if (history < 2 or steps <= history
            or abs(steps*model.dt-duration_s) > 1e-9
            or abs(history*model.dt-history_s) > 1e-9):
        raise ValueError('Segment/history must be whole steps, with segment longer than history')
    if not windows or len(windows) != len(metadata):
        raise ValueError('One source record is required per complete flight window')
    segments, records = [], []
    counts = []
    for window, source in zip(windows, metadata):
        total = len(window.applied_commands)
        if steps > total:
            raise ValueError('Residual segment exceeds the complete flight window')
        starts = list(range(0, total-steps+1, steps))
        if starts[-1] != total-steps:
            starts.append(total-steps)  # Include the tail; no padding or dropped samples.
        counts.append(len(starts))
        for index, start in enumerate(starts):
            end = start+steps
            if start == 0:
                if abs(source['initialization']['history_duration_s']-history_s) > 1e-6:
                    raise ValueError('Complete-flight initialization uses a different history duration')
                state, initialization = window.initial, source['initialization']
            else:
                if start <= history:
                    raise ValueError('Insufficient measured history for tail segment')
                # Full-window target row k is the state AFTER command k.
                past = slice(start-history-1, start)
                state, initialization = initialize(model,
                    np.arange(start-history, start+1)*model.dt,
                    window.tracking_positions[past], window.tracking_rotations[past],
                    window.cable_markers[past])
            segment = replace(window, initial=state,
                applied_commands=window.applied_commands[start:end].copy(),
                tracking_positions=window.tracking_positions[start:end].copy(),
                tracking_rotations=window.tracking_rotations[start:end].copy(),
                cable_markers=window.cable_markers[start:end].copy())
            segment.validate(model)
            segments.append(segment)
            records.append({**source, 'segment_index':index, 'parent_start_elapsed_s':source['start_elapsed_s'],
                'offset_steps':start, 'steps':steps,
                'start_elapsed_s':source['start_elapsed_s']+start*model.dt,
                'end_elapsed_s':source['start_elapsed_s']+end*model.dt,
                'start_source_s':source['start_source_s']+start*model.dt,
                'start_frame':source['start_frame']+start, 'end_frame':source['start_frame']+end,
                'initialization':initialization, 'purpose':'short-segment residual fitting'})
    if len(set(counts)) != 1:
        raise ValueError('Equal segment counts per take are required to preserve take weighting')
    for record, weight in zip(records, window_weights(segments)):
        record['objective_weight'] = weight
    return segments, records
