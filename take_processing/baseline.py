"""Load saved comparisons only when their source files and measured targets agree."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from take_processing.take_data import read_json, digest
from take_processing.library import verify_sources
from project_paths import project_path
from take_processing.paths import collection_root


@dataclass
class Baseline:
    time: np.ndarray
    position: np.ndarray
    rotation: np.ndarray
    markers: np.ndarray
    attachment: np.ndarray
    metadata: dict


def discover_baselines(take, root):
    source_hash = digest(take.path)
    found = []
    for path in sorted(project_path(root).glob('*/*.json'), reverse=True):
        if path.name == 'summary.json' or not path.with_suffix('.npz').is_file():
            continue
        try:
            meta = read_json(path)
            if meta.get('sources',{}).get('tracking',{}).get('sha256') == source_hash:
                found.append(path)
        except (ValueError, OSError):
            continue
    return found


def load_baseline(path, take):
    path = project_path(path)
    meta = read_json(path)
    if meta['sources']['tracking']['sha256'] != digest(take.path):
        raise ValueError('Baseline belongs to a different recording')
    verify_sources({'sources':meta['sources']}, root=collection_root(take.path.parent))
    if meta.get('arrays_sha256') is not None and digest(path.with_suffix('.npz')) != meta['arrays_sha256']:
        raise ValueError('Comparison array archive hash mismatch')
    with np.load(Path(path).with_suffix('.npz'), allow_pickle=False) as archive:
        a = {key:archive[key].copy() for key in archive.files}
    t = a['time_from_initial_s']
    n = len(t)
    dt = float(meta['model_config']['simulation']['timestep_s'])
    if n < 1 or not np.allclose(t, np.arange(1,n+1)*dt, rtol=0, atol=1e-8):
        raise ValueError('Invalid baseline output times')
    ids = np.flatnonzero(take.frame == meta['start_frame'])
    if len(ids)!=1:
        raise ValueError('Baseline start frame is unavailable')
    start = int(ids[0])
    if start+n >= len(take.time) or take.frame[start+n] != meta['end_frame']:
        raise ValueError('Baseline interval does not match take')
    if not np.allclose(take.time[start+1:start+n+1]-take.time[start],t,rtol=0,atol=1e-6):
        raise ValueError('Baseline and measurement timestamps disagree')
    if not np.isclose(take.source_time[start],meta['start_source_s'],rtol=0,atol=1e-8):
        raise ValueError('Baseline source clock does not match take')
    for key, target in (('measured_tracking_positions',take.position[start+1:start+n+1]),
                        ('measured_cable_markers',take.cable[start+1:start+n+1])):
        if not np.array_equal(a[key],target):
            raise ValueError('Saved measured targets do not match this take')
    for key, shape in (('predicted_tracking_positions',(n,3)),('predicted_tracking_rotations',(n,3,3)),
                       ('predicted_cable_markers',(n,10,3)),('initial_position_com',(3,)),
                       ('initial_rotation',(3,3))):
        if a[key].shape != shape or not np.isfinite(a[key]).all():
            raise ValueError(f'Invalid baseline array: {key}')
    rotations = np.concatenate([a['initial_rotation'][None],a['predicted_tracking_rotations']])
    if not np.allclose(rotations.transpose(0,2,1)@rotations,np.eye(3),atol=1e-6,rtol=0) or not np.allclose(np.linalg.det(rotations),1,atol=1e-6,rtol=0):
        raise ValueError('Invalid baseline rotations')
    config=meta['model_config']
    com=np.asarray(config['drone']['center_of_mass_tracking_m'])
    attachment=np.asarray(config['attachment']['position_tracking_m'])
    initial_position=a['initial_position_com']-a['initial_rotation']@com
    indices=np.cumsum(config['simulation']['segments_per_marker_interval']).astype(int)-1
    initial_markers=a['initial_free_positions'][indices]
    position=np.vstack([initial_position,a['predicted_tracking_positions']])
    markers=np.concatenate([initial_markers[None],a['predicted_cable_markers']])
    if attachment.shape!=(3,) or not np.isfinite(attachment).all() or not np.isfinite(markers).all():
        raise ValueError('Invalid baseline geometry')
    return Baseline(take.time[start]+np.r_[0.,t],position,rotations,markers,attachment,meta)
