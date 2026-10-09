"""Staged aerodynamic identification. Explicitly select the calibrated base artifact."""

from copy import deepcopy
import argparse
from datetime import datetime
import gc
import json
from pathlib import Path
import sys

from time import perf_counter

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

from simulation.tensor_model import TensorModel
from simulation.model_configuration import (load_model_configuration,
    configuration_digest, verify_configuration_sources)
from fitting.residual_network import AerodynamicResidual, load_residual, save_residual, SCHEMA
from fitting.prescribed_residual_objective import PrescribedResidualObjective
from fitting.batched_objective import BatchedObjective
from fitting.coupled_fit import window_weights
from fitting.residual_fit import fit_residual, hold_comparison
from fitting.residual_windows import prepare_residual_segments
from fitting.run_baseline import comparison_result
from fitting.run_fit import prepare_split, save, report_progress
from fitting.run_fit import SPLIT
from fitting.dataset import load_split
from fitting.artifacts import atomic_save
from project_paths import project_path, project_reference
from take_processing.take_data import digest
from take_processing.library import verify_sources
from take_processing.paths import flight_batches_path, select_recording_root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, help='Recording collection or flight_batches directory')
    parser.add_argument('--base-model', type=Path, required=True, help='Explicit calibrated physics or cable-stage candidate; never auto-promoted')
    parser.add_argument('--stage', choices=('cable','drone'), required=True)
    parser.add_argument('--split', type=Path, help='Same explicit recording split used for physical fitting')
    args = parser.parse_args()
    split,split_sources=load_split(args.split,SPLIT)
    root = select_recording_root(args.data, for_fitting=True)
    if root is None:
        return
    data_root = flight_batches_path(root)
    torch.set_num_threads(1)
    if not torch.cuda.is_available():raise RuntimeError('Residual fitting requires CUDA')
    settings_path=ROOT/'configs/fitting/residual.json'
    settings=json.loads(settings_path.read_text())
    settings_hash=digest(settings_path)
    torch.manual_seed(settings['training']['seed'])
    base_path=project_path(args.base_model)
    base,model_sources=load_model_configuration(base_path,with_sources=True)
    model_sources.update(split_sources)
    model_hash=configuration_digest(base)
    if base['residual']['status']=='fitted':
        model_sources['base_residual']={'path':project_reference(base['residual']['checkpoint']),
                                        'sha256':base['residual']['sha256']}
    if args.stage=='cable':
        if base['residual']['status'] != 'unfitted':
            raise ValueError('Cable stage starts from an explicitly calibrated physics-only artifact')
        network=AerodynamicResidual(settings['network']).to(device='cuda',dtype=torch.float64)
    else:
        if base['residual']['status'] != 'fitted':
            raise ValueError('Drone stage requires a frozen cable-stage candidate')
        network=load_residual(base['residual'],'cuda')
        if network.completed_stages != ['cable']:
            raise ValueError('Select the cable-stage checkpoint; do not refit both branches together')
    run_id=datetime.now().strftime('%Y%m%d-%H%M%S')
    output=project_path(ROOT/'runs/residual_fit'/run_id);output.mkdir(parents=True,exist_ok=False)
    started=perf_counter()
    def progress(row):
        row={**row,'total_elapsed_s':perf_counter()-started}
        if row.get('phase')=='gradient_check':
            save(output/('gradient_'+row['stage']+'.json'),row)
        report_progress(output,row)
    try:
        progress({'phase':'preparing','output':str(output),'base_model':str(base_path),
                  'maximum_updates':settings['training']['maximum_updates']})
        windows,metadata=prepare_split(data_root,TensorModel(base),split=split)
        segments,segment_metadata=prepare_residual_segments(TensorModel(base),windows['training'],metadata['training'],
            duration_s=settings['training']['segment_duration_s'],history_s=settings['training']['history_duration_s'])
        execution_hashes={str(p.relative_to(ROOT)):digest(p)
            for folder in ('fitting','simulation','controllers/full_state_controller','native')
            for p in (ROOT/folder).rglob('*') if p.suffix in ('.py','.cpp','.h','.hpp','.inc','.pyd')}
        def verify_execution():
            for name,sha in execution_hashes.items():
                if digest(ROOT/name)!=sha:
                    raise RuntimeError('Fitting implementation changed during execution: '+name)
            verify_configuration_sources(model_sources)
            if digest(settings_path)!=settings_hash:
                raise RuntimeError('Residual fitting settings changed during execution')
        save(output/'protocol.json',{'base_model':base,'base_model_sha256':model_hash,
            'base_model_hash_kind':'resolved_configuration','base_model_sources':model_sources,
            'settings':settings,'settings_sha256':settings_hash,'split':metadata,'stage':args.stage,
            'execution_hashes':execution_hashes,
            'training_segments':segment_metadata,
            'segment_coverage':'non-overlapping strides plus an end-aligned segment if needed; equal segments per take',
            'weighting':'equal maneuver families, then recording groups, then takes, then segments',
            'selection':'branch-owned short-segment training loss including magnitude penalty; continuous window comparisons only after selection',
            'cable_boundary':'measured tracking P/R; offline Hermite P and geodesic R; assumed initial twist',
            'loss_scope':'cable markers with measured boundary' if args.stage=='cable' else 'vehicle pose with frozen cable correction',
            'physical_parameters_frozen':True,'promoted':False})
        progress({'phase':'prepared','training_takes':len(windows['training']),'training_segments':len(segments),
                  'segment_steps':len(segments[0].applied_commands)})
        def checkpoint(value):
            value={**value,'schema':SCHEMA,'stage':args.stage,'specification':network.specification,
                   'base_model_sha256':model_hash,'settings_sha256':settings_hash}
            atomic_save(output/'training.pt',lambda temporary:torch.save(value,temporary))
        result=fit_residual(base,segments,network,settings['training'],stage=args.stage,progress=progress,checkpoint=checkpoint)
        save(output/'fit_result.json',result)
        # Release training graph cycles before allocating comparison graphs.
        gc.collect()
        verify_execution()
        weight_path=output/'residual.pt'
        weight_hash=save_residual(weight_path,network)
        candidate=deepcopy(base)
        candidate['residual']={'schema':SCHEMA,'status':'fitted','checkpoint':weight_path.relative_to(ROOT).as_posix(),'sha256':weight_hash}
        candidate_path=output/'model.json'
        save(candidate_path,candidate)
        records=[];losses={}
        if args.stage=='cable':
            conditional={}
            for role,group in windows.items():
                conditional[role]={}
                for label in ('before','after'):
                    probe=deepcopy(network)
                    if label=='before':
                        probe.specification.update(cable_enabled=False,drone_enabled=False)
                    objective=PrescribedResidualObjective(base,group,probe,0.)
                    conditional[role][label]=objective.loss()
            save(output/'prescribed_comparison.json',{'weighted_marker_loss':conditional,
                'normalization_m':.03,'reset':'once per complete selected window, never inside it'})
        for role,group in windows.items():
            predictions={};losses[role]={}
            for label,residual in (('before',None),('after',network)):
                progress({'phase':'comparison','role':role,'model':label})
                batch=BatchedObjective(base,group,window_weights(group),'cuda',graph=True,
                                       training=False,residual=residual)
                predictions[label]=batch.predict([0.]*len(batch.base_parameters))
                losses[role][label]=batch.evaluate([0.]*len(batch.base_parameters))[0]
                del batch
            for i,(window,meta) in enumerate(zip(group,metadata[role])):
                record={'take':meta['take'],'role':role,'maneuver':meta['maneuver']}
                for label,config in (('before',base),('after',candidate)):
                    p,q,R=(values[i] for values in predictions[label])
                    info,arrays=comparison_result(window,meta,p,R,q)
                    info.update(model_config=config,model_sha256=model_hash if label=='before' else digest(candidate_path),
                        model_hash_kind='resolved_configuration' if label=='before' else 'file',
                        base_model_sources=model_sources)
                    folder=output/label;folder.mkdir(exist_ok=True)
                    name=f"{role}_{i+1:02d}_{Path(meta['take']).stem}"
                    np.savez_compressed(folder/(name+'.npz'),**arrays)
                    save(folder/(name+'.json'),info);record[label]=info['metrics']
                records.append(record)
        progress({'phase':'hold_comparison','duration_s':30.})
        hold,hold_arrays=hold_comparison(base,network)
        save(output/'hold.json',hold)
        np.savez_compressed(output/'hold.npz',**hold_arrays)
        for group in metadata.values():
            for meta in group:verify_sources({'sources':meta['sources']})
        verify_execution()
        save(output/'summary.json',{'losses_without_penalty':losses,'takes':records,
            'hold':hold,'candidate_model':str(candidate_path),'weights':str(weight_path),'promoted':False})
        progress({'phase':'complete','candidate_model':str(candidate_path),'output':str(output),'promoted':False})
    except KeyboardInterrupt:
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
