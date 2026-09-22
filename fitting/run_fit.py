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
from fitting.prepare_window import prepare_window
from fitting.run_baseline import run_prepared, comparison_result
from take_processing.take_data import load_take, read_json, digest
from take_processing.library import verify_sources
from simulation.dynamics.coupled import CoupledDynamics


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
SPLIT = {role: [(motion,batch,f'{prefix}_whip_{number:03d}')
                for motion,batch,prefix,numbers in groups for number in numbers]
         for role,groups in SPLIT_GROUPS.items()}


def save(path, value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')
    temporary.replace(path)


def prepare_split(data_root, model, duration_s=1.7):
    windows={};metadata={};all_hashes=set()
    for role,rows in SPLIT.items():
        windows[role]=[];metadata[role]=[]
        for maneuver,batch,take in rows:
            path=(Path(data_root)/batch/'flight_take'/f'{take}.csv').resolve()
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
    parser.add_argument('--data',type=Path,default=ROOT.parent/'particle_filter_cable_project/data/flight_batches')
    parser.add_argument('--output',type=Path,default=ROOT/'runs/coupled_fit'/datetime.now().strftime('%Y%m%d-%H%M%S'))
    parser.add_argument('--iterations',type=int,default=30)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--device',choices=('cpu','cuda'),
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--no-compile',action='store_true',help='Use captured GPU fitting without compilation')
    args=parser.parse_args()
    torch.set_num_threads(1)
    model_path=ROOT/'models/physical_model.json';model_hash=digest(model_path)
    model=CoupledDynamics(model_path)
    windows,metadata=prepare_split(args.data,model)
    args.output.mkdir(parents=True,exist_ok=False)
    save(args.output/'protocol.json',{'base_model_sha256':model_hash,'base_model':model.config,
        'split':metadata,'parameters':dict(zip(NAMES,BOUNDS)),'iterations':args.iterations,
        'learning_rate':.05,'optimizer':'projected Adam in log multipliers',
        'workers':args.workers if args.device=='cpu' else 1,'device':args.device,
        'compiled':args.device=='cuda' and not args.no_compile,
        'position_scale_m':.02,'marker_scale_m':.03,'rotation_scale_rad':.05,
        'weighting':'equal motion types, then equal recording groups within type, then equal takes within group',
        'comparison_use':'after training-only candidate selection; historical prior exposure remains',
        'promoted':False})
    started=perf_counter()
    def progress(event):
        event=dict(event,elapsed_s=perf_counter()-started)
        if 'best_config' in event:
            save(args.output/'candidate_model.json',event.pop('best_config'))
        save(args.output/'progress.json',event)
        print(json.dumps(event),flush=True)
    try:
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
                                         'cuda',graph=True,training=False)
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
                    else:
                        comparison,arrays = run_prepared(windows[role][i],CoupledDynamics(config),meta)
                    comparison.update(role=role,maneuver=meta['maneuver'],model_config=config,
                        model_sha256=model_hash if label=='before' else digest(args.output/'candidate_model.json'))
                    name=f"{role}_{i+1:02d}_{Path(meta['take']).stem}"
                    np.savez_compressed(output/(name+'.npz'),**arrays)
                    save(output/(name+'.json'),comparison)
                    record[label]=comparison['metrics']
                summary.append(record)
                progress({'phase':'comparison_saved','take':meta['take'],'role':role,'before':record['before'],'after':record['after']})
        for rows in metadata.values():
            for meta in rows:verify_sources({'sources':meta['sources']})
        if digest(model_path)!=model_hash:raise RuntimeError('Active model changed during fit')
        save(args.output/'summary.json',summary)
        progress({'phase':'complete','output':str(args.output),'promoted':False})
    except Exception as exc:
        progress({'phase':'failed','error':str(exc)})
        raise


if __name__=='__main__':
    main()
