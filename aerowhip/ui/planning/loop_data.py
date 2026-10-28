"""What the Identification Loop window shows and runs: the trajectory files and their previews, the
measured envelope, the flown trajectory flights, whether a flight script is running, and the
detached jobs (optimise, preview, envelope, label) with their output in runs/planning/jobs/."""
from datetime import datetime
import json
import os
from pathlib import Path
import shlex
import subprocess

import numpy as np
from PySide6 import QtCore

from aerowhip.paths import ROOT
from aerowhip.planning.simulate import FLIGHT_STACK

PLANNING = ROOT/'runs'/'planning'
JOBS = PLANNING/'jobs'
PLANS = FLIGHT_STACK/'calibration_data'/'plans'
CONTAINER = '/workspace'
PYTHON = ROOT/'.venv'/'bin'/'python'
# a running flight script (the pattern skips editors and shells that merely name the file)
FLIGHT_PROCESS = r'python[0-9.]* .*acceleration_circle_cf2\.py'


def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def trajectories():
    """Excitation trajectory files under runs/planning/, newest first."""
    found = [p for p in PLANNING.glob('*.json') if not p.stem.endswith('_report') and 'excitation' in p.stem]
    return sorted(found, key=lambda p: p.stat().st_mtime, reverse=True)


def report_of(path):
    return read_json(path.with_name(path.stem+'_report.json'))


def preview_path(path):
    return path.with_name(path.stem+'_preview.npz')


def load_preview(path):
    p = preview_path(path)
    from aerowhip.online.state import STATE
    newest = max(path.stat().st_mtime, STATE.stat().st_mtime if STATE.exists() else 0.)
    if not p.exists() or p.stat().st_mtime < newest:    # older than the plan or the running model
        return None
    try:
        return dict(np.load(p))
    except (OSError, ValueError):
        return None


def next_name():
    numbers = [int(p.stem.split('_')[1]) for p in PLANNING.glob('excitation_[0-9][0-9][0-9].json')]
    return f'excitation_{max(numbers, default=0)+1:03d}'


def envelope_path(setup):
    return FLIGHT_STACK/'calibration_data'/'envelope'/f'{setup}.json'


def envelope(setup):
    return read_json(envelope_path(setup))


def clean_level(env):
    """Highest level flown cleanly (no failure, cable below the threshold), or None."""
    levels = [r['level'] for r in (env or {}).get('summary', []) if r.get('level') is not None and r['clean'] and not r['failed']]
    return max(levels, default=None)


def flown(setup):
    """Flight folders of trajectory flights, newest first."""
    base = FLIGHT_STACK/'calibration_data'/setup/'trajectory'
    return sorted((p for p in base.glob('*') if (p/'flight.jsonl').exists()), key=lambda p: p.name, reverse=True) if base.exists() else []


def flown_names(flights):
    return {(read_json(f/'trajectory.json') or {}).get('name') for f in flights[:50]}


def copied(path):
    target = PLANS/path.name
    return target.exists() and target.read_bytes() == path.read_bytes()


LIVE_VIEW = r'python[0-9.]* .*scripts/real_flight\.py'      # custom-real (the flight stack and its live view)


def live_view_running():
    return subprocess.run(['pgrep', '-f', LIVE_VIEW], capture_output=True).returncode == 0


def flight_running():
    return subprocess.run(['pgrep', '-f', FLIGHT_PROCESS], capture_output=True).returncode == 0


def pid_alive(pid):
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, ValueError):
        return False


class Job:
    """A detached process with its output in JOBS/<stamp>-<name>.log, its exit code in .exit and its
    shell's process id in .pid; found again from the files when the window is reopened."""
    def __init__(self, log):
        self.log = Path(log)
        self.name = self.log.stem.split('-', 2)[-1]

    @classmethod
    def start(cls, name, command, cwd=ROOT, env=''):
        JOBS.mkdir(parents=True, exist_ok=True)
        log = JOBS/f'{datetime.now():%Y%m%d-%H%M%S}-{name}.log'
        q = lambda p: shlex.quote(str(p))          # noqa: E731
        script = (f'echo $$ > {q(log.with_suffix(".pid"))}; cd {q(cwd)}; {env} nice -n 10 '
                  f'{" ".join(q(c) for c in command)} > {q(log)} 2>&1; echo $? > {q(log.with_suffix(".exit"))}')
        ok, _ = QtCore.QProcess.startDetached('/bin/sh', ['-c', script], str(cwd))
        return cls(log) if ok else None

    @classmethod
    def latest(cls):
        logs = sorted(JOBS.glob('*.log')) if JOBS.exists() else []
        return cls(logs[-1]) if logs else None

    @property
    def started(self):
        return self.log.with_suffix('.pid').stat().st_mtime if self.log.with_suffix('.pid').exists() else self.log.stat().st_mtime

    @property
    def exit_code(self):
        try:
            return int(self.log.with_suffix('.exit').read_text().strip())
        except (OSError, ValueError):
            return None

    @property
    def running(self):
        if self.exit_code is not None:
            return False
        try:
            return pid_alive(self.log.with_suffix('.pid').read_text().strip())
        except OSError:
            return True                            # just started: the shell has not written its id yet

    def output(self, chars=60000):
        try:
            text = self.log.read_text(errors='replace').splitlines()
        except OSError:
            return ''
        return '\n'.join(l for l in text if 'Warning' not in l)[-chars:]

    def stop(self):
        try:
            pid = self.log.with_suffix('.pid').read_text().strip()
        except OSError:
            return
        subprocess.run(['pkill', '-TERM', '-P', pid], check=False)


def flown_record(path, setup):
    """The newest real flight of the planned trajectory `path`, on the plan's timeline (t = 0 where its
    'excitation' part starts): drone positions and the cable markers as the live labeller sees them.
    Cached as runs/planning/<plan>_flown_<NNN>.npz. None if the plan has not been flown."""
    flights = [f for f in flown(setup) if (read_json(f/'trajectory.json') or {}).get('name') == path.stem]
    if not flights:
        return None
    folder = flights[0]
    cache = path.with_name(f'{path.stem}_flown_{folder.name}.npz')
    if cache.exists() and cache.stat().st_mtime >= (folder/'poses.jsonl').stat().st_mtime:
        return dict(np.load(cache), folder=str(folder))
    from aerowhip.online.stream import FlightStream
    from aerowhip.physics.model_configuration import DEFAULT_MODEL, fit_configuration, load_model_configuration
    cfg = fit_configuration(load_model_configuration(DEFAULT_MODEL))
    stream = FlightStream(folder, np.cumsum(cfg['cable']['marker_interval_lengths_m']), cfg['attachment']['position_tracking_m'],
                          keep_s=1e9, replay=True)
    stream.advance(np.inf)
    start = next((r['pose_stamp'] for r in stream.rows if r['phase'].startswith('excitation') and r.get('pose_stamp')), None)
    if start is None:
        return None
    stamps = np.array([p['stamp'] for p in stream.poses])
    position = np.array([p['position'] for p in stream.poses])
    labels = dict(stream.labels)
    markers = np.array([labels.get(s, np.full((10, 3), np.nan)) for s in stamps])
    record = dict(t=stamps-start, position=position, markers=markers)
    np.savez_compressed(cache, **record)
    return dict(record, folder=str(folder))
