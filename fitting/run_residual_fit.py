"""Run from PyCharm; settings live in models/residual_fit.json."""

from copy import deepcopy
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

from simulation.dynamics.coupled import CoupledDynamics
from simulation.dynamics.residuals import CoupledResidual, save_residual
from fitting.batched_objective import BatchedObjective
from fitting.coupled_fit import window_weights
from fitting.residual_fit import fit_residual, hold_comparison
from fitting.residual_windows import prepare_residual_segments
from fitting.run_baseline import comparison_result
from fitting.run_fit import prepare_split, save
from take_processing.take_data import digest
from take_processing.library import verify_sources


def main():
    torch.set_num_threads(1)
    if not torch.cuda.is_available():raise RuntimeError('Residual fitting requires CUDA')
    settings_path=ROOT/'models/residual_fit.json'
    settings=json.loads(settings_path.read_text())
    settings_hash=digest(settings_path)
    torch.manual_seed(settings['training']['seed'])
    model_path=ROOT/'models/physical_model.json';model_hash=digest(model_path)
    base=json.loads(model_path.read_text())
    if base.get('residual',{}).get('enabled',False):
        raise ValueError('This fit requires the physics-only baseline')
    run_id=datetime.now().strftime('%Y%m%d-%H%M%S')
    output=ROOT/'runs/residual_fit'/run_id;output.mkdir(parents=True,exist_ok=False)
    started=perf_counter()
    def progress(row):
        row={**row,'total_elapsed_s':perf_counter()-started}
        if row.get('phase')=='gradient_check':
            save(output/('gradient_'+row['stage']+'.json'),row)
        save(output/'progress.json',row)
        print(json.dumps(row),flush=True)
    try:
        progress({'phase':'preparing'})
        windows,metadata=prepare_split(ROOT.parent/'particle_filter_cable_project/data/flight_batches',CoupledDynamics(base))
        segments,segment_metadata=prepare_residual_segments(CoupledDynamics(base),windows['training'],metadata['training'],
            duration_s=settings['training']['segment_duration_s'],history_s=settings['training']['history_duration_s'])
        save(output/'protocol.json',{'base_model':base,'base_model_sha256':model_hash,
            'settings':settings,'settings_sha256':settings_hash,'split':metadata,
            'training_segments':segment_metadata,
            'segment_coverage':'non-overlapping strides plus an end-aligned segment if needed; equal segments per take',
            'weighting':'equal maneuver families, then recording groups, then takes, then segments',
            'selection':'short-segment training loss including magnitude penalty; full-flight comparison only after selection',
            'physical_parameters_frozen':True,'promoted':False})
        progress({'phase':'prepared','training_takes':len(windows['training']),'training_segments':len(segments),
                  'segment_steps':len(segments[0].applied_commands)})
        network=CoupledResidual(settings['network']).to(device='cuda',dtype=torch.float64)
        def checkpoint(value):
            temporary=output/'training.tmp';torch.save(value,temporary);temporary.replace(output/'training.pt')
        result=fit_residual(base,segments,network,settings['training'],progress=progress,checkpoint=checkpoint)
        save(output/'fit_result.json',result)
        # Release training graph cycles before allocating comparison graphs.
        gc.collect()
        weight_path=ROOT/'models'/f'residual_{run_id}.pt'
        weight_hash=save_residual(weight_path,network)
        candidate=deepcopy(base)
        candidate['residual']={'enabled':True,'checkpoint':weight_path.relative_to(ROOT).as_posix(),'sha256':weight_hash}
        candidate_path=ROOT/'models'/f'residual_{run_id}.json'
        save(candidate_path,candidate)
        records=[];losses={}
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
                    info.update(model_config=config,model_sha256=model_hash if label=='before' else digest(candidate_path))
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
        if digest(model_path)!=model_hash or digest(settings_path)!=settings_hash:
            raise RuntimeError('Model or settings changed during residual fitting')
        save(output/'summary.json',{'losses_without_penalty':losses,'takes':records,
            'hold':hold,'candidate_model':str(candidate_path),'weights':str(weight_path),'promoted':False})
        progress({'phase':'complete','candidate_model':str(candidate_path),'output':str(output),'promoted':False})
    except Exception as exc:
        progress({'phase':'failed','error':str(exc)});raise


if __name__=='__main__':
    main()
