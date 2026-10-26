"""Fit/refit one coupled physical model: material identification, then coupled response.

Fits EI/Cb/drag from measured-root cable motion, then vehicle response with the
entire cable configuration frozen. Same native plant. No automatic promotion.
Residual identification remains a separately assessed stage after physical fitting.
"""
import argparse
from datetime import datetime
from pathlib import Path
import subprocess
import sys
from time import perf_counter

from aerowhip.paths import ROOT
if __package__ in (None,''):
    sys.path.insert(0,str(ROOT))

from aerowhip.fitting.cable_fit import CableFitSettings, fit_cable, prepare_windows
from aerowhip.fitting.dataset import load_split
from aerowhip.fitting.run_fit import SPLIT, prepare_split, report_progress, save
from aerowhip.physics.tensor_model import TensorModel
from aerowhip.physics.model_configuration import DEFAULT_FIT_MODEL, load_model_configuration, verify_configuration_sources, configuration_digest
from aerowhip.takes.paths import select_recording_root
from aerowhip.takes.take_data import digest
from aerowhip.paths import project_path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path)
    parser.add_argument('--base-model',type=Path,default=DEFAULT_FIT_MODEL)
    parser.add_argument('--split',type=Path,help='JSON with training/comparison [maneuver,batch,take] rows')
    parser.add_argument('--output',type=Path,default=ROOT/'runs/refit'/datetime.now().strftime('%Y%m%d-%H%M%S-%f'))
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--iterations',type=int,default=30)
    parser.add_argument('--material-refinements',type=int,default=2)
    parser.add_argument('--prepare-only',action='store_true',help='Validate and save selection without optimizing')
    args=parser.parse_args()
    settings=CableFitSettings(workers=args.workers,refinements=args.material_refinements)
    settings.validate()
    if args.iterations<1:
        parser.error('Positive coupled iteration count required')
    split,split_sources=load_split(args.split,SPLIT)
    base,sources=load_model_configuration(args.base_model,with_sources=True)
    sources.update(split_sources)
    if base['residual']['status']!='unfitted':
        raise ValueError('Refit the physics-only parent, then reassess residuals; fitted residual input is not allowed')
    root=select_recording_root(args.data,for_fitting=True)
    if root is None:
        return
    out=project_path(args.output)
    out.mkdir(parents=True,exist_ok=False)
    started=perf_counter()
    def progress(row):
        # Detailed material summaries already live with that stage's artifacts.
        row={key:value for key,value in row.items() if key!='summary'}
        report_progress(out,dict(row,total_elapsed_s=perf_counter()-started))
    def verify():
        verify_configuration_sources(sources)
    save(out/'split.json',split)
    save(out/'base_model.json',base)
    save(out/'protocol.json',dict(base_model=base,base_model_sha256=configuration_digest(base),
         sources=sources,split=split,workers=args.workers,material_refinements=args.material_refinements,
         coupled_iterations=args.iterations,prepare_only=args.prepare_only,
         stages=['measured-root EI/Cb/external drag, all-marker loss',
                 'controller gains, vehicle pose loss, all cable parameters frozen'],
         fixed='drone geometry/mass/inertia, motors, clamp, GJ/Ct, integration and constraints',
         parameter_bounds='multipliers around each stage input; not hardware identification limits',
         comparison_use='after training-only selection; historical exposure remains',promoted=False))
    try:
        verify()
        if args.prepare_only:
            model=TensorModel(base)
            material=prepare_windows(root,model,settings,split=split)
            _,coupled=prepare_split(root,model,split=split)
            verify()
            save(out/'prepared.json',dict(material_windows=material,coupled_windows=coupled))
            progress(dict(phase='prepared',output=str(out),optimization=False))
            return
        progress(dict(phase='material_start',output=str(out)))
        material=fit_cable(root,out/'base_model.json',settings,output_root=out/'material',
             comparison_root=out/'material_comparisons',split=split,
             cancelled=lambda:(out/'STOP').exists(),progress=lambda r:progress(dict(stage='material',**r)))
        verify()
        material_model=material/'candidate_model.json'
        progress(dict(phase='coupled_start',material_model=str(material_model)))
        command=[sys.executable,'-u','-m','aerowhip.fitting.run_fit','--data',str(root),
                 '--base-model',str(material_model),'--split',str(out/'split.json'),
                 '--output',str(out/'coupled'),'--device','native','--workers',str(args.workers),
                 '--iterations',str(args.iterations)]
        # Child maintains immutable protocol/events, comparisons and failure evidence.
        # Stream output to the invoking terminal; this is not a silent background fit.
        completed=subprocess.run(command,cwd=ROOT)
        if completed.returncode:
            if (out/'STOP').exists() or (out/'coupled/STOP').exists():
                raise InterruptedError('Physical refit stopped; partial artifacts retained')
            raise RuntimeError(f'Coupled refit exited {completed.returncode}; inspect coupled/events.jsonl')
        verify()
        candidate=out/'coupled/candidate_model.json'
        if not candidate.is_file() or not (out/'coupled/summary.json').is_file():
            raise RuntimeError('Coupled stage did not finish its saved evaluation')
        selected=load_model_configuration(candidate)
        cable_stage=load_model_configuration(material_model)
        if selected['cable'] != cable_stage['cable']:
            raise RuntimeError('Vehicle stage changed frozen cable parameters')
        save(out/'result.json',dict(status='complete',candidate_model=str(candidate),
             candidate_sha256=digest(candidate),material_result=str(material/'summary.json'),
             coupled_result=str(out/'coupled/fit_result.json'),comparison=str(out/'coupled/summary.json'),
             elapsed_s=perf_counter()-started,promoted=False,cable_ownership_verified=True,
             caveat='Bounded calibration, not convergence, unseen hardware validation or residual fitting.'))
        progress(dict(phase='complete',candidate_model=str(candidate),promoted=False))
    except BaseException as exc:
        progress(dict(phase='stopped' if isinstance(exc,(KeyboardInterrupt,InterruptedError)) else 'failed',error=str(exc),promoted=False))
        raise


if __name__=='__main__':main()
