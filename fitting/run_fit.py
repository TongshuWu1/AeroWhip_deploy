"""Approved mixed-maneuver fit; saves a candidate without model promotion."""
import argparse
from datetime import datetime
from dataclasses import replace
import json
from pathlib import Path
import sys

from time import perf_counter
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

from fitting.coupled_fit import fit_coupled, window_weights, NAMES, BOUNDS
from fitting.artifacts import atomic_save
from fitting.prepare_window import prepare_window
from fitting.run_baseline import run_prepared, comparison_result
from take_processing.take_data import load_take, read_json, digest
from take_processing.library import verify_sources
from simulation.tensor_model import TensorModel
from simulation.model_configuration import (DEFAULT_FIT_MODEL, load_model_configuration,
    configuration_digest, verify_configuration_sources)
from project_paths import project_path
from take_processing.paths import flight_batches_path, select_recording_root
from fitting.dataset import load_split, validate_split


SPLIT_GROUPS = {
    'training': [
        ('vertical_plane','M3-session1-four-whips','M3_001', (1,2,4)),
        ('vertical_plane','M3-session3-four-whips','M3_003', (1,2,3,4)),
        ('vertical_plane','M4-session1-four-whips','M4_001', (1,2,3,4)),
        ('vertical_plane','M4-session3-four-whips','M4_003', (1,2,3)),
        ('horizontal_side','M5-curved-side-session1-four-whips','M5_side_001', (1,2,3,4)),
        ('horizontal_side','M6-horizontal-session1-four-whips','M6_side_001', (1,4)),
    ],
    'comparison': [
        ('vertical_plane','M3-session2-four-whips','M3_002', (3,4)),
        ('vertical_plane','M4-session2-four-whips','M4_002', (1,2,3,4)),
        ('horizontal_side','M5-curved-side-session2-four-whips','M5_side_002', (1,2,3,4)),
        ('horizontal_side','M6-horizontal-session2-four-whips','M6_side_002', (1,2,3)),
    ],
}
FULL_SPLIT = {role: [(motion,batch,f'{prefix}_whip_{number:03d}')
                for motion,batch,prefix,numbers in groups for number in numbers]
         for role,groups in SPLIT_GROUPS.items()}

# One take from each maneuver M3--M6, chosen by recording order, not fit error.
# Keep the historical catalog and all comparison flights; unused training takes
# are NOT a new held-out set (older physical fits already used them).
TRAINING_SELECTION = (
    ('M3-session1-four-whips', 'M3_001_whip_001'),
    ('M4-session1-four-whips', 'M4_001_whip_001'),
    ('M5-curved-side-session1-four-whips', 'M5_side_001_whip_001'),
    ('M6-horizontal-session1-four-whips', 'M6_side_001_whip_001'),
)
SPLIT = {
    'training': [row for row in FULL_SPLIT['training'] if row[1:] in TRAINING_SELECTION],
    'comparison': list(FULL_SPLIT['comparison']),
}
if (len(set(TRAINING_SELECTION)) != len(TRAINING_SELECTION)
        or {row[1:] for row in SPLIT['training']} != set(TRAINING_SELECTION)):
    raise ValueError('Fitting selection must name unique training-catalog takes')


def save(path, value):
    """Replace JSON atomically; retry brief Windows sharing/access conflicts.

    Exhausted retries still raise: model and result saves must not be lost
    silently. Only the replaceable progress display is non-critical below.
    """
    text=json.dumps(value,indent=2,allow_nan=False)
    atomic_save(path,lambda temporary:temporary.write_text(text,encoding='utf-8'))


def report_progress(output, event):
    """Keep the event history before updating the best-effort status display."""
    event=dict(event)
    if 'best_config' in event:
        save(output/'candidate_model.json',event.pop('best_config'))
    line=json.dumps(event,allow_nan=False)
    with (output/'events.jsonl').open('a',encoding='utf-8') as handle:
        handle.write(line+'\n')
    print(line,flush=True)
    try:
        save(output/'progress.json',event)
    except OSError as exc:
        print(f'Warning: progress display could not be updated; event retained in events.jsonl: {exc}',
              file=sys.stderr,flush=True)


def prepare_split(data_root, model, duration_s=1.7, *, split=None):
    data_root = flight_batches_path(data_root)
    split = validate_split(SPLIT if split is None else split)
    windows={};metadata={};all_hashes=set()
    for role,rows in split.items():
        windows[role]=[];metadata[role]=[]
        for maneuver,batch,take in rows:
            path=data_root/batch/'flight_take'/f'{take}.csv'
            data=load_take(path)
            segment=read_json(path.with_name('experiment_'+path.stem+'.metadata.json'))['segment']
            start=segment['onset_s']-data.alignment['offset_s']-data.source_time[0]-.1
            window,meta=prepare_window(path,model,start_s=start,duration_s=duration_s)
            window=replace(window,maneuver=maneuver)
            sha=meta['sources']['tracking']['sha256']
            if sha in all_hashes:raise ValueError('Duplicate source recording in split')
            all_hashes.add(sha)
            meta.update(role=role,maneuver=maneuver,purpose='approved mixed-maneuver calibration')
            windows[role].append(window);metadata[role].append(meta)
        for meta,weight in zip(metadata[role],window_weights(windows[role])):
            meta['objective_weight']=weight
    if {w.recording_group for w in windows['training']}&{w.recording_group for w in windows['comparison']}:
        raise ValueError('Recording groups overlap')
    return windows,metadata


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,help='Recording collection or flight_batches directory')
    parser.add_argument('--base-model',type=Path,default=DEFAULT_FIT_MODEL,
                        help='Explicit physics-only model to fit/refit; never overlaid with defaults')
    parser.add_argument('--split',type=Path,help='JSON training/comparison recording lists')
    parser.add_argument('--output',type=Path,default=ROOT/'runs/coupled_fit'/datetime.now().strftime('%Y%m%d-%H%M%S'))
    parser.add_argument('--iterations',type=int,default=30)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--device',choices=('native','cpu','cuda'),default='native',
                        help='Native CPU finite differences; tensor CPU/CUDA remain diagnostic alternatives')
    parser.add_argument('--no-compile',action='store_true',help='Use captured GPU fitting without compilation')
    args=parser.parse_args()
    split,split_sources=load_split(args.split,SPLIT)
    if args.iterations<1 or args.workers<1:
        parser.error('Positive iterations and workers required')
    root = select_recording_root(args.data, for_fitting=True)
    if root is None:
        return
    data_root = flight_batches_path(root)
    args.output = project_path(args.output)
    torch.set_num_threads(1)
    base,model_sources=load_model_configuration(args.base_model,with_sources=True)
    if base['residual']['status']!='unfitted':
        raise ValueError('Physical refitting requires a physics-only artifact; do not refit around learned drag')
    model_sources.update(split_sources)
    model_hash=configuration_digest(base)
    model=TensorModel(base)
    print(f'Preparing {len(split["training"])} fitting and {len(split["comparison"])} comparison takes...',flush=True)
    windows,metadata=prepare_split(data_root,model,split=split)
    args.output.mkdir(parents=True,exist_ok=False)
    execution_hashes={str(p.relative_to(ROOT)):digest(p)
        for folder in ('fitting','simulation','controllers/full_state_controller','native')
        for p in (ROOT/folder).rglob('*') if p.suffix in ('.py','.cpp','.h','.hpp','.inc','.pyd','.so')}
    save(args.output/'protocol.json',{'base_model_sha256':model_hash,'base_model':model.config,
        'base_model_hash_kind':'resolved_configuration','base_model_sources':model_sources,
        'split':metadata,'parameters':dict(zip(NAMES,BOUNDS)),'iterations':args.iterations,
        'fixed_bending_stiffness_n_m2':base['cable']['EI_n_m2'],
        'execution_hashes':execution_hashes,
        'learning_rate':.05,'optimizer':'projected Adam in log multipliers',
        'workers':args.workers if args.device!='cuda' else 1,'device':args.device,
        'derivatives':'central finite differences, epsilon=1e-5 in log scales; initial/selected checks at 1e-5 and 5e-6' if args.device=='native' else 'automatic differentiation',
        'bounds_reference':'multipliers relative to the explicitly selected base model, not absolute material/hardware limits',
        'compiled':args.device=='cuda' and not args.no_compile,
        'position_scale_m':.02,'rotation_scale_rad':.05,
        'loss_scope':'vehicle position and attitude only; cable markers are evaluation metrics',
        'fixed_cable':base['cable'],
        'weighting':'equal motion types, then equal recording groups within type, then equal takes within group',
        'comparison_use':'after training-only candidate selection; historical prior exposure remains',
        'promoted':False})
    started=perf_counter()
    def progress(event):
        verify_configuration_sources(model_sources)
        for name,sha in execution_hashes.items():
            if digest(ROOT/name)!=sha:
                raise RuntimeError('Fitting implementation changed during execution: '+name)
        if event['phase'] not in ('failed','stopped','complete') and ((args.output/'STOP').exists()
                or (args.output.parent/'STOP').exists()):
            raise InterruptedError('Refit STOP requested; partial artifacts retained, no promotion')
        report_progress(args.output,dict(event,elapsed_s=perf_counter()-started))
    try:
        progress({'phase':'starting','iterations':args.iterations,'output':str(args.output),
                  'parameters':list(NAMES),'fixed_bending_stiffness_n_m2':base['cable']['EI_n_m2']})
        result=fit_coupled(model.config,windows['training'],windows['comparison'],
            iterations=args.iterations,learning_rate=.05,workers=args.workers,progress=progress,
            device=args.device,compile_physics=not args.no_compile)
        save(args.output/'candidate_model.json',result['config'])
        save(args.output/'fit_result.json',result)
        summary=[]
        for role,rows in metadata.items():
            predictions = {}
            if args.device == 'cuda':
                from fitting.batched_objective import BatchedObjective
                progress({'phase':'comparison_replay','role':role,'windows':len(rows),'device':'cuda'})
                # Short-lived inference needs no compiler/backward setup. Reuse
                # the prepared states and capture once for both parameter sets.
                batch = BatchedObjective(model.config,windows[role],window_weights(windows[role]),
                                         'cuda',graph=True,training=False,loss_scope='vehicle')
                for label,values in (('before',[0.]*len(NAMES)),
                                     ('after',np.log([result['scales'][name] for name in NAMES]))):
                    predictions[label] = batch.predict(values)
                del batch
            for i,meta in enumerate(rows):
                record={'take':meta['take'],'role':role,'maneuver':meta['maneuver']}
                for label,config in (('before',model.config),('after',result['config'])):
                    output=args.output/label;output.mkdir(exist_ok=True)
                    if predictions:
                        positions,markers,rotations = (values[i] for values in predictions[label])
                        comparison,arrays = comparison_result(windows[role][i],meta,positions,rotations,markers)
                    elif args.device == 'native':
                        from fitting.native_objective import rollout
                        _,(positions,markers,rotations)=rollout(config,windows[role][i],[0.]*len(NAMES))
                        comparison,arrays=comparison_result(windows[role][i],meta,positions,rotations,markers)
                    else:
                        comparison,arrays = run_prepared(windows[role][i],TensorModel(config),meta)
                    comparison.update(role=role,maneuver=meta['maneuver'],model_config=config,
                        model_sha256=model_hash if label=='before' else digest(args.output/'candidate_model.json'),
                        model_hash_kind='resolved_configuration' if label=='before' else 'file',
                        base_model_sources=model_sources)
                    name=f"{role}_{i+1:02d}_{Path(meta['take']).stem}"
                    np.savez_compressed(output/(name+'.npz'),**arrays)
                    save(output/(name+'.json'),comparison)
                    record[label]=comparison['metrics']
                summary.append(record)
                progress({'phase':'comparison_saved','take':meta['take'],'role':role,'before':record['before'],'after':record['after']})
        for rows in metadata.values():
            for meta in rows:verify_sources({'sources':meta['sources']})
        verify_configuration_sources(model_sources)
        save(args.output/'summary.json',summary)
        progress({'phase':'complete','output':str(args.output),'promoted':False})
    except (KeyboardInterrupt, InterruptedError):
        try:
            progress({'phase':'stopped','promoted':False})
        except Exception as report_error:
            print(f'Could not record stopped status: {report_error}',file=sys.stderr,flush=True)
        raise
    except Exception as exc:
        try:
            progress({'phase':'failed','error':str(exc)})
        except Exception as report_error:
            print(f'Could not record failure status: {report_error}',file=sys.stderr,flush=True)
        raise


if __name__=='__main__':
    main()
