"""Machine-readable progress of a fit, one JSON object per line, for the Model Fit window.

A fit script writes `<result>.progress.jsonl` next to its result file:

    {"kind": "start", "stage": ..., "parameters": [...], "x0": [...], "unit": "m", "pid": 1234, "time": ...}
    {"kind": "plan", "phases": ["loading flights", ...], "evaluations": [2, ...], "time": ...}
    {"kind": "phase", "index": 0, "label": "loading flights", "time": ...}
    {"kind": "evaluation", "n": 1, "phase": 2, "x": [...], "value": 0.123, "time": ...}
    {"kind": "phase_done", "index": 2, "label": "delay 0 ms", "value": 0.0563, "x": [...], "time": ...}
    {"kind": "iteration", "n": 0, "best": 0.12, "x": [...], "time": ...}
    {"kind": "note", "text": "...", "time": ...}
    {"kind": "done", "result": "<result file>", "best": [...], "value": ..., "time": ...}
    {"kind": "failed", "error": "...", "time": ...}

Lines are flushed as written, so the window (or `tail -f`) sees a fit started anywhere. The plan
gives each phase an upper bound on its residual evaluations (a least-squares phase can converge
earlier), which the window turns into a progress bar and an estimate of the time left.
"""
import json
import math
import os
from pathlib import Path
import time


def progress_path(result):
    result = Path(result)
    return result.with_name(result.stem+'.progress.jsonl')


class Progress:
    def __init__(self, result, *, stage, parameters, x0, unit=''):
        self.path = progress_path(result)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.path.open('w', buffering=1)
        self.evaluations = 0
        self.current = None
        self.write(kind='start', stage=stage, parameters=list(parameters), x0=[float(v) for v in x0], unit=unit,
                   result=str(result), pid=os.getpid())

    def write(self, **values):
        values['time'] = time.time()
        self.file.write(json.dumps(values, allow_nan=True)+'\n')

    def plan(self, phases, evaluations):
        """Phase labels and the most residual evaluations each can take."""
        self.labels = list(phases)
        self.write(kind='plan', phases=self.labels, evaluations=[int(n) for n in evaluations])

    def phase(self, index):
        self.current = int(index)
        self.write(kind='phase', index=self.current, label=self.labels[self.current])

    def phase_done(self, value, x=()):
        self.write(kind='phase_done', index=self.current, label=self.labels[self.current],
                   value=float(value) if math.isfinite(value) else None, x=[float(v) for v in x])

    def evaluation(self, x, value):
        self.evaluations += 1
        self.write(kind='evaluation', n=self.evaluations, phase=self.current, x=[float(v) for v in x],
                   value=float(value) if math.isfinite(value) else None)

    def iteration(self, n, best, x):
        self.write(kind='iteration', n=int(n), best=float(best), x=[float(v) for v in x])

    def note(self, text):
        self.write(kind='note', text=str(text))

    def done(self, best, value):
        self.write(kind='done', result=str(self.path).replace('.progress.jsonl', '.json'),
                   best=[float(v) for v in best], value=float(value))
        self.file.close()

    def failed(self, error):
        self.write(kind='failed', error=str(error))
        self.file.close()


def read(path):
    """All events of a progress file (a partial last line is skipped)."""
    events = []
    for line in Path(path).read_text().splitlines():
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            break
    return events
