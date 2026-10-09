"""Residual-off conditional EI/Cb/external-drag identification on recorded motion.

Measured attachment motion drives the existing C++ cable solver. Comparison
takes never choose parameters; no active model is replaced.
"""
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
import multiprocessing as mp
from itertools import product
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import numpy as np
import torch

from fitting.calibration import _compare_prepared, _identities, save_comparison
from fitting.artifacts import atomic_save
from fitting.coupled_fit import window_weights
from fitting.recorded_replay import prepare_recorded_replay
from fitting.run_fit import SPLIT
from fitting.dataset import validate_split
from project_paths import ROOT, project_path
from simulation.tensor_model import TensorModel
from simulation.model_configuration import load_model_configuration
from take_processing.library import verify_sources
from take_processing.paths import flight_batches_path
from take_processing.take_data import digest, load_take, read_json


@dataclass(frozen=True)
class CableFitSettings:
    workers: int = 4
    refinements: int = 2
    minimum_scale: float = .1
    maximum_scale: float = 10.
    minimum_drag_scale: float = .25
    maximum_drag_scale: float = 4.
    early_duration_s: float = 1.7
    late_duration_s: float = 2.

    def validate(self):
        if (type(self.workers) is not int or not 1 <= self.workers <= 32
                or type(self.refinements) is not int or not 0 <= self.refinements <= 6):
            raise ValueError('Use 1–32 workers and 0–6 refinement rounds')
        lo, hi = self.minimum_scale, self.maximum_scale
        if not np.isfinite([lo, hi]).all() or not 0 < lo < 1 < hi:
            raise ValueError('EI/Cb scale bounds must be finite, positive and bracket 1')
        lo,hi = self.minimum_drag_scale,self.maximum_drag_scale
        if not np.isfinite([lo,hi]).all() or not 0 < lo < 1 < hi:
            raise ValueError('External-drag scale bounds must be finite, positive and bracket 1')
        for value in (self.early_duration_s, self.late_duration_s):
            if not np.isfinite(value) or value < .01 or abs(round(value/.01)*.01-value) > 1e-9:
                raise ValueError('Window durations must be positive whole 10-ms steps')


def _write(path, value, *, exclusive=False):
    text = json.dumps(value, indent=2, allow_nan=False)+'\n'
    if exclusive:
        with path.open('x', encoding='utf-8') as stream:
            stream.write(text)
    else:
        atomic_save(path,lambda temporary:temporary.write_text(text,encoding='utf-8'))


def _check(cancelled):
    if cancelled is not None and cancelled():
        raise InterruptedError('Cable fit stopped; active models unchanged')


def prepare_windows(root, model, settings, *, progress=None, cancelled=None, split=None):
    """Fail closed on missing takes/frames/alignment in the shared active split."""
    settings.validate()
    split = validate_split(SPLIT if split is None else split)
    batches = flight_batches_path(root)
    windows, hashes, groups = [], set(), {}
    for role, rows in split.items():
        groups[role] = set()
        for maneuver, batch, name in rows:
            _check(cancelled)
            path = batches/batch/'flight_take'/f'{name}.csv'
            take = load_take(path)
            if not take.alignment_applied:
                raise ValueError(f'{name}: verified saved clock alignment is required for window selection')
            segment = read_json(path.with_name('experiment_'+name+'.metadata.json'))['segment']
            offset = float(take.alignment['offset_s'])+float(take.source_time[0])
            onset = float(segment['onset_s'])-offset
            commands = np.atleast_1d(np.genfromtxt(
                path.with_name('experiment_'+name+'.commands.csv'),
                delimiter=',', names=True, encoding='utf-8-sig'))
            motion = np.column_stack([commands['cmd_'+key] for key in ('vx','vy','vz','ax','ay','az')])
            times = commands['time_s']
            if (not np.isfinite(motion).all() or not np.isfinite(times).all()
                    or np.any(np.diff(times) <= 0) or np.any(commands['cmd_valid'] != 1)):
                raise ValueError(f'{name}: invalid command log for late-window selection')
            moving = np.any(np.abs(motion) > 1e-8, axis=1)
            if not np.any(moving):
                raise ValueError(f'{name}: no nonzero command V/A for late-window selection')
            late = float(times[moving][-1])-offset+.3
            for phase, start, duration in (('early', onset-.1, settings.early_duration_s),
                                            ('late', late, settings.late_duration_s)):
                _check(cancelled)
                replay = prepare_recorded_replay(path, model, start_s=start, duration_s=duration,
                                                 require_commands=False)
                meta = replay.metadata
                sha = meta['sources']['tracking']['sha256']
                if phase == 'early':
                    if sha in hashes:
                        raise ValueError('Duplicate source recording in fitting split')
                    hashes.add(sha)
                groups[role].add(meta['recording_group'])
                windows.append(dict(id=len(windows), role=role, maneuver=maneuver, phase=phase,
                    path=str(path), start_s=start, duration_s=duration, metadata=meta))
                if progress:
                    progress(dict(phase='preparation', completed=len(windows), total=2*sum(map(len, split.values())),
                                  message=f'{name} · {phase}'))
    if groups['training'] & groups['comparison']:
        raise ValueError('Recording groups overlap between fitting and comparison')
    for role in split:
        selected = [w for w in windows if w['role'] == role]
        weights = window_weights([SimpleNamespace(recording_group=w['metadata']['recording_group'],
                                                 maneuver=w['maneuver']) for w in selected])
        for window, weight in zip(selected, weights):
            window['weight'] = weight
    return windows


def search_material(objective, settings):
    """Joint 3x3x3 log grid and bounded refinements, not convergence proof."""
    settings.validate()
    scores = {}

    def evaluate(ei, cb, drag):
        # Canonicalize log/exp roundoff; reuse the exact baseline and grid center.
        key = tuple(float(f'{v:.14g}') for v in (ei,cb,drag))
        if key not in scores:
            value = float(objective(*key))
            if not math.isfinite(value):
                raise FloatingPointError('Nonfinite fitting objective')
            scores[key] = value

    evaluate(1.,1.,1.)
    material_limits=(math.log(settings.minimum_scale), math.log(settings.maximum_scale))
    limits=[material_limits,material_limits,
            (math.log(settings.minimum_drag_scale),math.log(settings.maximum_drag_scale))]
    bounds=list(limits)
    for _ in range(settings.refinements+1):
        axes=[np.linspace(lo,hi,3) for lo,hi in bounds]
        for values in product(*axes):
            evaluate(*(math.exp(float(value)) for value in values))
        best = min(scores, key=lambda s:(scores[s],sum(abs(math.log(x)) for x in s)))
        for i in range(3):
            radius=(bounds[i][1]-bounds[i][0])/4
            center=math.log(best[i])
            bounds[i]=(max(limits[i][0],center-radius),min(limits[i][1],center+radius))
    best = min(scores, key=lambda s:(scores[s],sum(abs(math.log(x)) for x in s)))
    return best, scores


_CACHE = {}
_BASE = _STOP = None


def _initialize_worker(base, stop):
    global _BASE, _STOP, _CACHE
    torch.set_num_threads(1)
    _BASE, _STOP, _CACHE = base, stop, {}


@torch.no_grad()
def _evaluate_window(window, model_path, identities):
    _check(_STOP.is_set)
    verify_sources(window['metadata'])
    config = load_model_configuration(model_path)
    unchanged = deepcopy(config)
    for key in ('EI_n_m2', 'Cb_n_m2_s', 'external_drag_s_inv'):
        unchanged['cable'][key] = _BASE['cable'][key]
    if unchanged != _BASE:
        raise ValueError('Cached preparation permits only EI, Cb and external-drag changes')
    model = TensorModel(config, device='cpu')
    if window['id'] not in _CACHE:
        _CACHE[window['id']] = prepare_recorded_replay(window['path'], model,
            start_s=window['start_s'], duration_s=window['duration_s'], require_commands=False)
    replay = _CACHE[window['id']]
    if replay.metadata != window['metadata']:
        raise RuntimeError('Prepared recording changed during fitting')
    return _compare_prepared(model, replay, identities, cancelled=_STOP.is_set)


def _aggregate(rows, windows):
    return {key: math.sqrt(sum(w['weight']*rows[w['id']][0]['metrics'][key]**2 for w in windows))
            for key in ('cable_marker_rmse_m', 'tip_rmse_m')}


def fit_cable(root, model_path, settings=None, *, output_root=None, comparison_root=None,
              progress=None, cancelled=None, split=None):
    """Fit EI/Cb/external drag jointly with fixed GJ/Ct; no promotion."""
    settings = settings or CableFitSettings()
    split = validate_split(SPLIT if split is None else split)
    settings.validate()
    _check(cancelled)
    model_path = project_path(model_path)
    base = load_model_configuration(model_path)
    if base['residual']['status'] == 'fitted':
        raise ValueError('Select an explicitly residual-disabled physical model')
    for key in ('EI_n_m2', 'Cb_n_m2_s', 'external_drag_s_inv'):
        if not math.isfinite(base['cable'][key]) or base['cable'][key] <= 0:
            raise ValueError(f'{key} must be positive for log-scale identification')
    identities = _identities(model_path, base)
    for relative in ('fitting/cable_fit.py', 'fitting/run_fit.py', 'fitting/coupled_fit.py','fitting/dataset.py',
                     'take_processing/paths.py', 'take_processing/library.py'):
        identities[relative] = dict(path=str(ROOT/relative), sha256=digest(ROOT/relative))
    if load_model_configuration(model_path) != base:
        raise RuntimeError('Model changed during loading')
    parent = project_path(output_root) if output_root is not None else ROOT/'runs/cable_fit'
    folder = parent/datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    folder.mkdir(parents=True, exist_ok=False)
    started = perf_counter()

    def emit(event):
        event = dict(event, elapsed_s=perf_counter()-started,
                     time_utc=datetime.now(timezone.utc).isoformat(), run=str(folder))
        with (folder/'events.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(event, allow_nan=False)+'\n')
        _write(folder/'status.json', event)
        if progress:
            progress(event)

    def verify():
        _check(cancelled)
        for identity in identities.values():
            if digest(Path(identity['path'])) != identity['sha256']:
                raise RuntimeError('Code, model or binary changed during fitting')
        for window in windows:
            verify_sources(window['metadata'])

    protocol = dict(schema='material_cable_fit_v3', settings=asdict(settings), identities=identities,
        parameters={'fitted': ['cable.EI_n_m2','cable.Cb_n_m2_s','cable.external_drag_s_inv'], 'fixed': 'all other fields including GJ, Ct, drone/controller/motors'},
        loss='mean squared Euclidean marker-position error [m^2]; no tip double counting',
        weighting='equal maneuver families, recording groups, takes, early/late windows',
        window_selection='early: onset -0.1 s; late: last logged |V/A| >1e-8 +0.3 s; native 100 Hz',
        method='3x3x3 joint log grid plus baseline; bounded local refinement in EI/Cb/external drag',
        torsion='GJ/Ct fixed assumptions; zero initial relative material angles, not observed twist',
        comparison_use='only after selection; historical prior exposure, not pristine test data',
        stopping='fixed search budget, not established convergence', residual_enabled=False,
        physics_backend='native_cpu', device='cpu', dtype='float64', promoted=False)
    _write(folder/'protocol.json', protocol, exclusive=True)
    _write(folder/'base_model.json', base, exclusive=True)
    stop = mp.get_context('spawn').Event()
    try:
        take_count = sum(map(len, split.values()))
        emit(dict(phase='preparation', completed=0, total=2*take_count, message=f'Checking {take_count} selected takes'))
        torch.set_num_threads(1)
        windows = prepare_windows(root, TensorModel(base), settings, progress=emit, cancelled=cancelled,split=split)
        _write(folder/'windows.json', windows, exclusive=True)
        verify()
        fitting = [w for w in windows if w['role'] == 'training']
        comparison = [w for w in windows if w['role'] == 'comparison']
        cache, retained = {}, {}
        trials = folder/'trials'
        trials.mkdir()
        with ProcessPoolExecutor(max_workers=settings.workers, mp_context=mp.get_context('spawn'),
                                 initializer=_initialize_worker, initargs=(base, stop)) as pool:
            def evaluate(ei, cb, drag, *, role='training'):
                _check(cancelled)
                key = (float(ei), float(cb), float(drag), role)
                if key in cache:
                    return cache[key]
                trial = trials/f'{len(cache):03d}'
                trial.mkdir()
                config = deepcopy(base)
                config['cable']['EI_n_m2'] *= ei
                config['cable']['Cb_n_m2_s'] *= cb
                config['cable']['external_drag_s_inv'] *= drag
                path = trial/'model.json'
                _write(path, config, exclusive=True)
                trial_ids = deepcopy(identities)
                trial_ids['base_model'] = trial_ids['model']
                trial_ids['model'] = dict(path=str(path), sha256=digest(path))
                selected = fitting if role == 'training' else comparison
                label = f'{role} · EI ×{ei:.5g} · Cb ×{cb:.5g} · external drag ×{drag:.5g}'
                emit(dict(phase='evaluation', completed=0, total=len(selected), message=label))
                pending = {pool.submit(_evaluate_window, w, path, trial_ids): w for w in selected}
                rows = {}
                try:
                    while pending:
                        _check(cancelled)
                        done, _ = wait(pending, timeout=.1, return_when=FIRST_COMPLETED)
                        for future in done:
                            window = pending.pop(future)
                            rows[window['id']] = future.result()
                            emit(dict(phase='evaluation', completed=len(rows), total=len(selected), message=label))
                except BaseException:
                    stop.set()
                    for future in pending:
                        future.cancel()
                    raise
                _check(cancelled)
                metrics = _aggregate(rows, selected)
                result = dict(cb_scale=cb, ei_scale=ei, drag_scale=drag, role=role, metrics=metrics,
                    loss_m2=metrics['cable_marker_rmse_m']**2, model_path=str(path),
                    per_window=[dict(id=w['id'], metrics=rows[w['id']][0]['metrics'],
                                     numerics=rows[w['id']][0]['numerics']) for w in selected])
                _write(trial/'result.json', result, exclusive=True)
                cache[key] = result
                if role == 'training':
                    if ei == 1. and cb == 1. and drag == 1.:
                        retained['before'] = rows
                    previous = retained.get('best_result')
                    distance = lambda r: sum(abs(math.log(r[k+'_scale'])) for k in ('ei','cb','drag'))
                    if previous is None or (result['loss_m2'],distance(result)) < (
                            previous['loss_m2'],distance(previous)):
                        retained['after'], retained['best_result'] = rows, result
                elif role == 'comparison':
                    retained[('comparison',ei,cb,drag)] = rows
                emit(dict(phase='trial_complete', completed=len(selected), total=len(selected),
                          message=label, **{k: result[k] for k in ('loss_m2', 'metrics')}))
                return result

            best, scores = search_material(lambda ei,cb,drag: evaluate(ei,cb,drag)['loss_m2'], settings)
            before_comparison = evaluate(1.,1.,1.,role='comparison')
            after_comparison = evaluate(*best,role='comparison')
        verify()
        selected_result = cache[(*best,'training')]
        overlays = []
        for role, selected in (('training', fitting), ('comparison', comparison)):
            for label, pair in (('before',(1.,1.,1.)),('after',best)):
                rows = retained[label] if role == 'training' else retained[('comparison',*pair)]
                for window in selected:
                    _check(cancelled)
                    meta, arrays = rows[window['id']]
                    meta = dict(meta, cable_fit_run=str(folder), cable_fit_variant=label,
                                cable_fit_role=role, cable_fit_phase=window['phase'],
                                purpose='cable-only physical identification; no model promotion')
                    overlay = save_comparison(meta, arrays, output_root=comparison_root)
                    overlays.append(dict(id=window['id'], role=role, variant=label, path=str(overlay)))
        verify()
        summary = dict(best_EI_scale=best[0],best_Cb_scale=best[1],best_drag_scale=best[2],
            EI_n_m2=base['cable']['EI_n_m2']*best[0], Cb_n_m2_s=base['cable']['Cb_n_m2_s']*best[1],
            fixed_GJ_n_m2=base['cable']['GJ_n_m2'], fixed_Ct_n_m2_s=base['cable']['Ct_n_m2_s'],
            external_drag_s_inv=base['cable']['external_drag_s_inv']*best[2],
            boundary_best=any(np.isclose(value,bound,atol=0,rtol=1e-12)
                              for value,bounds in zip(best,[(settings.minimum_scale,settings.maximum_scale)]*2+
                                  [(settings.minimum_drag_scale,settings.maximum_drag_scale)]) for bound in bounds),
            training_before=cache[(1.,1.,1.,'training')]['metrics'],
            training_after=selected_result['metrics'],
            comparison_before=before_comparison['metrics'], comparison_after=after_comparison['metrics'],
            material_search=[dict(EI_scale=s[0],Cb_scale=s[1],drag_scale=s[2],loss_m2=v) for s,v in sorted(scores.items())],
            identifiability='conditional fit with assumed GJ/Ct and initial material angles; uniqueness not established',
            overlays=overlays, promoted=False, trials=len(cache))
        _write(folder/'summary.json', summary, exclusive=True)
        # No candidate filename appears until search, comparison and provenance checks complete.
        _write(folder/'candidate_model.json', read_json(Path(selected_result['model_path'])), exclusive=True)
        artifacts = {str(p.relative_to(folder)): digest(p) for p in folder.rglob('*')
                     if p.is_file() and p.name not in ('events.jsonl', 'status.json')}
        _write(folder/'artifacts.json', artifacts, exclusive=True)
        emit(dict(phase='complete', message='Candidate saved; active model unchanged',
                  completed=1, total=1, summary=summary))
        return folder
    except InterruptedError:
        stop.set()
        emit(dict(phase='stopped', message='Stopped; partial evidence retained, no model promotion'))
        raise
    except BaseException as exc:
        stop.set()
        emit(dict(phase='failed', message=str(exc)))
        raise
