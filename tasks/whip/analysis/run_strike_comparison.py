"""PyCharm: rescore nine saved replays; no model construction or training."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import shutil
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
if __package__ in (None, ''):
    sys.path.insert(0, str(ROOT))
from tasks.whip.reach import ReachConfiguration, window_distance
from tasks.whip.analysis import ARCHIVE_ROOT

SOURCE_RUN = ARCHIVE_ROOT/'runs/exploration_comparison/20260922-014754-089375'
STRIKE = dict(sigma_m=.10, bonus_weight=1., speed_reference_mps=5.)
ARMS = ('current_smooth', 'preparation_strike', 'swing_candidates')
SEEDS = (0, 1, 2)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def components(distance, forward_speed, *, sigma_m, bonus_weight, speed_reference_mps):
    """Same-time soft targetward strike shaping; no hit or whip classification."""
    if (not np.isfinite([sigma_m, bonus_weight, speed_reference_mps]).all()
            or sigma_m <= 0 or bonus_weight < 0 or speed_reference_mps <= 0):
        raise ValueError('Require positive distance/speed scales and nonnegative bonus weight')
    if (distance.ndim != 1 or distance.shape != forward_speed.shape
            or not np.isfinite(distance).all() or not np.isfinite(forward_speed).all()
            or np.any(distance < 0) or distance[0] <= 0):
        raise ValueError('Require finite matching histories and positive initial distance')
    progress = np.clip(1-distance/distance[0], 0, 1)
    proximity = np.exp(-.5*(distance/sigma_m)**2)
    speed = np.maximum(forward_speed, 0)
    bonus = bonus_weight*proximity*speed/(speed+speed_reference_mps)
    return progress, bonus, progress+bonus


def forward_crossings(time, tip, velocity, target, normal, width, height, eligible):
    """Forward intersections of the piecewise-linear sampled tip path.

    The half-open rule f_left<=0<f_right counts an exact sampled plane point
    once. For on-plane plateaus, the preceding off-plane sample must be on the
    negative side, excluding touches from either side and initial departures.
    A crossing is placed at the final plane sample. Rectangle inclusion uses
    a scale-aware floating-point tolerance, not a physical hit tolerance.
    Eligibility uses interpolated time.
    Stored node velocities are interpolated separately from the segment slope;
    neither is a contact-force calculation or an exact continuous-time event.
    """
    relative = tip-target
    forward = relative @ normal
    side = np.array([-normal[1], normal[0], 0.])
    rows = []
    tolerance = 16*np.finfo(float).eps*max(1., float(np.max(np.abs(time))), *map(abs, eligible))
    for index in np.flatnonzero((forward[:-1] <= 0) & (forward[1:] > 0)):
        if forward[index] == 0:
            preceding = np.flatnonzero(forward[:index] != 0)
            if not len(preceding) or forward[preceding[-1]] > 0:
                continue
        fraction = -forward[index]/(forward[index+1]-forward[index])
        event = time[index]+fraction*(time[index+1]-time[index])
        if not eligible[0]-tolerance <= event <= eligible[1]+tolerance:
            continue
        point = tip[index]+fraction*(tip[index+1]-tip[index])
        speed = velocity[index]+fraction*(velocity[index+1]-velocity[index])
        lateral = float((point-target) @ side)
        vertical = float(point[2]-target[2])
        outside_side = max(abs(lateral)-width/2, 0.)
        outside_vertical = max(abs(vertical)-height/2, 0.)
        geometry_tolerance = 16*np.finfo(float).eps*max(1., float(np.max(np.abs(point))),
            float(np.max(np.abs(target))), width, height)
        if outside_side <= geometry_tolerance: outside_side = 0.
        if outside_vertical <= geometry_tolerance: outside_vertical = 0.
        rows.append(dict(time_s=float(event), position_world_m=point.tolist(),
            interpolated_velocity_world_mps=speed.tolist(),
            forward_speed_mps=float(speed @ normal),
            segment_forward_speed_mps=float((forward[index+1]-forward[index])/(time[index+1]-time[index])),
            within_window=bool(outside_side == 0 and outside_vertical == 0),
            lateral_miss_m=outside_side, vertical_miss_m=outside_vertical,
            left_sample_index=int(index), interpolation_fraction=float(fraction)))
    return rows


def verification():
    d = np.array([1.19, .10, 0.])
    v = np.array([0., 5., 1.])
    p, b, score = components(d, v, **STRIKE)
    np.testing.assert_allclose(score, p+np.exp(-.5*(d/.1)**2)*np.maximum(v,0)/(np.maximum(v,0)+5), rtol=1e-14)
    assert score[1] > score[2]  # Deliberately preserve the agreed near-miss tradeoff.
    assert components(d, -np.ones(3), **STRIKE)[1].max() == 0
    assert components(np.array([1.19,.70]), np.array([0.,5.]), **STRIKE)[1][1] < 1e-10
    assert score.max() < p.max()+b.max()  # Cannot combine different event maxima.
    time = np.array([0., .1, .2])
    target = np.array([0.,0.,1.75]); normal = np.array([1.,0.,0.])
    tip = np.array([[-.1,0.,1.75],[0.,0.,1.75],[.1,0.,1.75]])
    velocity = np.tile([1.,0.,0.],(3,1))
    crossing = forward_crossings(time,tip,velocity,target,normal,.8,.6,(0.,.2))
    assert len(crossing)==1 and crossing[0]['within_window'] and crossing[0]['time_s']==.1
    assert not forward_crossings(time,tip[::-1],-velocity,target,normal,.8,.6,(0.,.2))
    touch = tip.copy(); touch[-1,0]=-.1
    assert not forward_crossings(time,touch,velocity,target,normal,.8,.6,(0.,.2))
    assert not forward_crossings(time,touch*np.array([-1.,1.,1.]),velocity,target,normal,.8,.6,(0.,.2))
    departure = tip.copy(); departure[:,0] = [0.,0.,.1]
    assert not forward_crossings(time,departure,velocity,target,normal,.8,.6,(0.,.2))
    edge = tip.copy(); edge[:,2] = 1.45
    edge_crossing = forward_crossings(time,edge,velocity,target,normal,.8,.6,(0.,.2))
    assert len(edge_crossing)==1 and edge_crossing[0]['within_window'] and edge_crossing[0]['vertical_miss_m']==0
    below = tip.copy(); below[:,2]=1.40
    missed = forward_crossings(time,below,velocity,target,normal,.8,.6,(0.,.2))
    assert len(missed)==1 and not missed[0]['within_window']
    np.testing.assert_allclose(missed[0]['vertical_miss_m'], .05, atol=1e-14)
    assert not forward_crossings(time,tip,velocity,target,normal,.8,.6,(.11,.2))
    try:
        components(np.array([1.,np.nan]), np.zeros(2), **STRIKE)
    except ValueError:
        pass
    else:
        raise AssertionError('Nonfinite input was not rejected')
    return dict(formula=True, same_time_selection=True, bounded_distance_speed_bonus=True,
        fast_near_miss_tradeoff_preserved=True, backward_speed_no_bonus=True,
        forward_interpolation=True, exact_plane_no_duplicate=True, touch_return_not_crossing=True,
        outside_window_not_hit=True, crossing_time_window=True, nonfinite_rejected=True,
        lower_window_edge_included=True, initial_plane_departure_excluded=True,
        physics_rollouts=0, optimizer_updates=0)


def plot_comparison(rows, output):
    labels = [row['name'].replace('preparation_strike','Preparation').replace(
        'current_smooth','Smooth').replace('swing_candidates','Swing').replace('_seed_',' / seed ') for row in rows]
    y = np.arange(len(rows))
    fig, axes = plt.subplots(1,3,figsize=(13,6),sharey=True,layout='constrained')
    axes[0].scatter([r['old_rank'] for r in rows],y,label='Previous reward',marker='o',color='#888888')
    axes[0].scatter([r['new_rank'] for r in rows],y,label='Candidate reward',marker='x',color='#237eb2')
    axes[0].set_xlabel('Rank (1 = highest)'); axes[0].set_xticks(range(1,10)); axes[0].legend(fontsize=8)
    for axis,key,title in ((axes[1],'distance_m','Window distance (m)'),(axes[2],'forward_speed_mps','Forward speed (m/s)')):
        axis.scatter([r['closest_event'][key] for r in rows],y,label='Closest approach',color='#888888')
        axis.scatter([r['new_event'][key] for r in rows],y,label='New scored event',marker='x',color='#237eb2')
        axis.set_xlabel(title); axis.legend(fontsize=8); axis.grid(axis='x',alpha=.2)
    axes[0].set_yticks(y,labels); axes[0].invert_yaxis()
    crossings = sum(row['forward_window_crossing_count'] for row in rows)
    fig.suptitle(f'Nine retained replays: same-time strike rescoring\nNo new simulation or learning; {crossings} observed forward finite-window crossings')
    fig.savefig(output/'comparison.png',dpi=145)
    plt.close(fig)


def main():
    torch.set_num_threads(2)
    checks = verification()
    source_manifest = json.loads((SOURCE_RUN/'artifacts.json').read_text())
    def verified(relative):
        path = SOURCE_RUN/relative
        if source_manifest.get(str(relative)) != digest(path):
            # Original Windows manifest keys use backslashes.
            if source_manifest.get(str(relative).replace('/', '\\')) != digest(path):
                raise ValueError(f'Source artifact hash mismatch: {relative}')
        return path
    protocol_path = verified(Path('protocol.json'))
    raw = json.loads(protocol_path.read_text())['configuration']
    target = np.array(raw['experiment']['target_position_world_m'])
    width, height = raw['reach']['window_width_m'],raw['reach']['window_height_m']
    delay = raw['_study_delay_s']
    eligible = (delay,delay+raw['trajectory']['duration_s']+raw['reward']['tip_window_s'])
    protected = {str(path.relative_to(ROOT)):digest(path) for name in ('simulation','training','trajectories','tasks/whip','models','configs')
        for path in (ROOT/name).rglob('*') if path.is_file() and '__pycache__' not in path.parts and 'analysis' not in path.parts
        and path.suffix in ('.py','.json','.pt','.toml')}
    output = ROOT/'runs/strike_comparison'/datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-%f')
    output.mkdir(parents=True,exist_ok=False)
    print(f'Rescoring nine saved replays: {output}',flush=True)
    shutil.copy2(__file__,output/Path(__file__).name)
    shutil.copy2(ROOT/'tasks/whip/reach.py',output/'source_reach.py')
    shutil.copy2(protocol_path,output/'source_protocol.json')
    write_json(output/'protocol.json',dict(source_run=str(SOURCE_RUN),source_manifest_sha256=digest(SOURCE_RUN/'artifacts.json'),
        strike=STRIKE,eligible_time_s=eligible,protected_sha256=protected,
        versions=dict(python=platform.python_version(),numpy=np.__version__,torch=str(torch.__version__)),
        selection='Exactly the nine previously selected closest-window candidates; not reranking all 2304 proposals.',
        formula='max_t [clip(1-d(t)/d(0),0,1) + beta*exp(-d(t)^2/(2*sigma^2))*max(n.v_tip(t),0)/(max(n.v_tip(t),0)+v_ref)] - saved cost',
        crossing='Piecewise-linear observed forward plane crossing; finite-window bounds checked at interpolated point. No contact validation.',
        physics_rollouts=0,optimizer_updates=0,production_reward_changed=False))
    rows=[]
    for arm in ARMS:
        for seed in SEEDS:
            name=f'{arm}_seed_{seed}'; relative=Path(name)
            paths={key:verified(relative/file) for key,file in [('summary','summary.json'),('rollout','rollout.npz'),('episodes','episodes.npz')]}
            saved=json.loads(paths['summary'].read_text()); selected=saved['selected_index']
            with np.load(paths['rollout'],allow_pickle=False) as data:
                arrays={key:data[key] for key in data.files}
            with np.load(paths['episodes'],allow_pickle=False) as records:
                np.testing.assert_array_equal(arrays['action'],records['actions'][selected])
                valid=bool(records['valid'][selected]); old_return=float(records['returns'][selected]); cost=float(records['cost'][selected])
            if not valid or not arrays['state_valid'].all():
                raise ValueError(f'{name}: this selected-replay comparison requires saved valid episodes')
            for key in ('time_s','cable_positions','cable_velocities','drone_positions','drone_rotations','commands'):
                if not np.isfinite(arrays[key]).all(): raise ValueError(f'{name}: nonfinite {key}')
            time=arrays['time_s']; np.testing.assert_allclose(time,np.arange(351)*.01,rtol=0,atol=1e-12)
            steps=round(raw['trajectory']['duration_s']/.01)
            reconstructed_cost=raw['return']['acceleration_cost_weight']*np.mean(np.sum(
                (arrays['commands'][:steps,6:9]/raw['trajectory']['max_acceleration_mps2'])**2,axis=-1))
            np.testing.assert_allclose(cost,reconstructed_cost,rtol=1e-12,atol=1e-14)
            tip,velocity=arrays['cable_positions'][:,-1],arrays['cable_velocities'][:,-1]
            start=arrays['drone_positions'][0]
            normal=target-start; normal[2]=0.; normal/=np.linalg.norm(normal)
            tensor=lambda x:torch.as_tensor(x,dtype=torch.float64)
            distance=window_distance(tensor(tip)[None],tensor(start)[None],tensor(target)[None],
                ReachConfiguration(width,height,raw['reach']['whip_bonus_weight']))[0].numpy()
            forward=velocity@normal
            progress,bonus,history=components(distance,forward,**STRIKE)
            tolerance=16*np.finfo(float).eps*max(1.,float(np.max(np.abs(time))))
            permitted=(time>=eligible[0]-tolerance)&(time<=eligible[1]+tolerance)
            new_index=int(np.argmax(np.where(permitted,history,-np.inf)))
            closest=int(np.argmin(np.where(permitted,distance,np.inf)))
            np.testing.assert_allclose(distance[closest],saved['selected']['closest_window_distance_m'],rtol=1e-12,atol=1e-12)
            def event(index):
                return dict(index=index,time_s=float(time[index]),distance_m=float(distance[index]),
                    center_distance_m=float(np.linalg.norm(tip[index]-target)),position_world_m=tip[index].tolist(),
                    forward_speed_mps=float(forward[index]),tip_speed_mps=float(np.linalg.norm(velocity[index])),
                    progress=float(progress[index]),strike_bonus=float(bonus[index]))
            crossing=forward_crossings(time,tip,velocity,target,normal,width,height,eligible)
            row=dict(name=name,arm=arm,seed=seed,source_candidate_index=selected,
                inputs={key:dict(path=str(path),sha256=digest(path)) for key,path in paths.items()},
                valid=valid,acceleration_cost=cost,old_return=old_return,new_return=float(history[new_index]-cost),
                closest_event=event(closest),new_event=event(new_index),forward_plane_crossings=crossing,
                forward_window_crossing_count=sum(item['within_window'] for item in crossing))
            rows.append(row)
            with (output/f'{name}.npz').open('xb') as stream:
                np.savez_compressed(stream,time_s=time,distance_m=distance,forward_speed_mps=forward,
                    progress=progress,strike_bonus=bonus,score_history=history,eligible=permitted)
    for rank,row in enumerate(sorted(rows,key=lambda r:r['old_return'],reverse=True),1): row['old_rank']=rank
    rows.sort(key=lambda r:r['new_return'],reverse=True)
    for rank,row in enumerate(rows,1): row['new_rank']=rank
    plot_comparison(rows,output)
    if any(digest(ROOT/name)!=expected for name,expected in protected.items()):
        raise RuntimeError('Protected production input/source changed during analysis')
    checks.update(source_hashes=True,selected_action_identity=True,command_cost_reconstruction=True,
        previous_closest_distance_agreement=True,protected_files_unchanged=len(protected))
    write_json(output/'verification.json',checks)
    write_json(output/'comparison.json',dict(cases=rows,physics_rollouts=0,optimizer_updates=0,
        trajectories_with_forward_window_crossing=sum(r['forward_window_crossing_count']>0 for r in rows),
        limitation='Selected replay rescore only; soft near misses can outrank slower hits. Not a learning or whip-classification result.'))
    write_json(output/'artifacts.json',{str(path.relative_to(output)):digest(path) for path in output.rglob('*') if path.is_file()})
    for row in rows:
        print(json.dumps({key:row[key] for key in ('name','old_rank','new_rank','old_return','new_return','new_event','forward_window_crossing_count')}),flush=True)
    print(f'Finished: {output}',flush=True)
    return output


if __name__=='__main__':
    main()
