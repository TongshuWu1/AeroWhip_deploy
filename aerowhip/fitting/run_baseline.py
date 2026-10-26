"""Run unchanged coupled-model comparisons and save numerical experiment output."""
import argparse
from copy import deepcopy
from dataclasses import fields
from datetime import datetime
import json
from pathlib import Path
import sys

import numpy as np
import torch

from aerowhip.paths import ROOT
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

from aerowhip.fitting.prepare_window import prepare_window
from aerowhip.takes.take_data import load_take, read_json
from aerowhip.takes.library import verify_sources
from aerowhip.physics.tensor_model import TensorModel
from aerowhip.physics.model_configuration import DEFAULT_FIT_MODEL, load_model_configuration, configuration_digest, verify_configuration_sources
from aerowhip.control.tensor_controller import GeometricController
from aerowhip.paths import project_path
from aerowhip.takes.paths import resolve_recording_path


def run(path, model, duration, start=None):
    path = resolve_recording_path(path)
    if start is None:
        take = load_take(path)
        segment = read_json(path.with_name('experiment_'+path.stem+'.metadata.json')).get('segment',{})
        if not take.alignment_applied or 'onset_s' not in segment:
            raise ValueError("Supply --start when there is no saved maneuver onset")
        start = segment['onset_s']-take.alignment['offset_s']-take.source_time[0]-.1
    window, metadata = prepare_window(path,model,start_s=start,duration_s=duration)
    result = run_prepared(window, model, metadata)
    verify_sources({"sources":metadata['sources']})
    return result


def run_prepared(window, model, metadata):
    """Replay an already aligned/initialized window without re-reading its take."""
    controller = GeometricController(model)
    state = window.initial
    positions, rotations, markers = [], [], []
    with torch.no_grad():
        for command in window.applied_commands:
            state = model.step(state,command,controller)
            if not all(bool(torch.isfinite(getattr(state,f.name)).all()) for f in fields(state)):
                raise FloatingPointError('Nonfinite baseline state')
            p,_ = model.tracking_pose(state);q,_=model.cable_state(state)
            positions.append(p.numpy().copy());rotations.append(state.rotation.numpy().copy())
            markers.append(q[model.marker_indices[1:]].numpy().copy())
    positions, rotations, markers = map(np.asarray,(positions,rotations,markers))
    return comparison_result(window, metadata, positions, rotations, markers)


def comparison_result(window, metadata, positions, rotations, markers):
    """Identical metrics and replay arrays for CPU and batched GPU predictions."""
    metadata = deepcopy(metadata)
    if not all(np.isfinite(value).all() for value in (positions, rotations, markers)):
        raise FloatingPointError('Nonfinite replay predictions')
    angle = np.arccos(np.clip((np.einsum('nij,nij->n',rotations,window.tracking_rotations)-1)/2,-1,1))
    norm_rms = lambda error: float(np.sqrt(np.mean(np.sum(error**2,axis=-1))))
    metadata['metrics'] = {"tracking_position_rmse_m":norm_rms(positions-window.tracking_positions),
        "cable_marker_rmse_m":norm_rms(markers-window.cable_markers),
        "tip_rmse_m":norm_rms(markers[:,-1]-window.cable_markers[:,-1]),
        "attitude_rmse_deg":float(np.rad2deg(np.sqrt(np.mean(angle**2))))}
    arrays = {"time_from_initial_s":np.arange(1,len(positions)+1)*window.timestep_s,
        "applied_commands":window.applied_commands,"measured_tracking_positions":window.tracking_positions,
        "measured_tracking_rotations":window.tracking_rotations,"measured_cable_markers":window.cable_markers,
        "predicted_tracking_positions":positions,"predicted_tracking_rotations":rotations,
        "predicted_cable_markers":markers}
    arrays.update({"initial_"+f.name:getattr(window.initial,f.name).numpy() for f in fields(window.initial)})
    return metadata, arrays


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('takes',type=Path,nargs='+')
    parser.add_argument('--duration',type=float,default=2.5)
    parser.add_argument('--start',type=float,help='Elapsed tracking time; otherwise 0.1 s before saved onset')
    parser.add_argument('--output',type=Path,default=ROOT/'runs'/'coupled_baseline'/datetime.now().strftime('%Y%m%d-%H%M%S'))
    args=parser.parse_args()
    args.output = project_path(args.output)
    torch.set_num_threads(1)
    base,model_sources=load_model_configuration(DEFAULT_FIT_MODEL,with_sources=True)
    config_hash=configuration_digest(base)
    model=TensorModel(base)
    args.output.mkdir(parents=True,exist_ok=False)
    results=[]
    for i,path in enumerate(args.takes):
        path=resolve_recording_path(path)
        print('Running',path.stem,flush=True)
        try:
            metadata,arrays=run(path,model,args.duration,args.start)
            metadata['model_sha256']=config_hash
            metadata['model_hash_kind']='resolved_configuration'
            metadata['model_sources']=model_sources
            metadata['model_config']=model.config
            file=f'{i+1:02d}_{path.stem}'
            np.savez_compressed(args.output/(file+'.npz'),**arrays)
            (args.output/(file+'.json')).write_text(json.dumps(metadata,indent=2,allow_nan=False),encoding='utf-8')
            results.append({'take':str(path),'metrics':metadata['metrics'],'initialization':metadata['initialization']})
            print(json.dumps(results[-1]),flush=True)
        except (ValueError,OSError,FloatingPointError,RuntimeError) as exc:
            results.append({'take':str(path),'error':str(exc)})
            print('Rejected:',str(exc),flush=True)
    verify_configuration_sources(model_sources)
    (args.output/'summary.json').write_text(json.dumps(results,indent=2,allow_nan=False),encoding='utf-8')
    print('Output:',args.output,flush=True)


if __name__=='__main__':
    main()
