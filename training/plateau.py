"""Conservative stage-aware plateau heuristic; NOT proof of convergence/success."""
from copy import deepcopy
import math


def validate_plateau(settings):
    c=settings.get('plateau_detection')
    if c is None:return
    if settings.get('algorithm')!='ppo' or settings.get('training_stage') not in (1,2,3,4):
        raise ValueError('Plateau detection requires explicit staged PPO')
    keys={'mode','minimum_collections','window_collections','patience_checks',
          'check_interval_collections','require_entropy_floor','minimum_gain'}
    if not isinstance(c,dict) or set(c)!=keys or c['mode'] not in ('warn','stop'):
        raise ValueError('Invalid plateau configuration/mode')
    for key in ('minimum_collections','window_collections','patience_checks','check_interval_collections'):
        if type(c[key]) is not int or c[key]<1:raise ValueError('Invalid plateau '+key)
    if c['minimum_collections']<40:raise ValueError('Plateau stopping must allow at least40 complete collections')
    if c['window_collections']<2:raise ValueError('Plateau windows require at least2 collections')
    if type(c['require_entropy_floor']) is not bool:raise ValueError('require_entropy_floor must be boolean')
    if settings['training_episodes']<c['minimum_collections']*settings['parallel_environments']:
        raise ValueError('Budget must allow the minimum number of full training collections')
    if c['require_entropy_floor'] and 'hold_fraction' not in settings['ppo']['entropy']:
        raise ValueError('Plateau entropy guard requires an episode-budget schedule')
    gains=c['minimum_gain']
    if not isinstance(gains,dict) or set(gains)!={'return','quality','rate','containment_m'}:
        raise ValueError('Provide explicit plateau gain thresholds and units')
    for v in gains.values():
        if isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or v<=0:
            raise ValueError('Plateau gains must be positive and finite')


def metric_categories(stage):
    common={'raw_return':'return','validity':'rate','whip_quality':'quality','containment':'containment_m'}
    if stage>=2:common['hit_rate']='rate'
    if stage==2:common['delivery_quality']='quality'
    if stage==3:common['precision_quality']='quality'
    if stage==4:
        common.update(joint_rate='rate',drone_hover_rate='rate',cable_settle_rate='rate',
                      recovery_quality='quality',cable_quality='quality')
    return common


def collection_metrics(stage,returns,valid,score,workspace):
    """Equal episode ALL-lane means; failed/missing-event positive credit is zero.

    Raw failed returns are retained. Containment is negative executed-prefix
    equal-episode RMS excess (m), a soft tradeoff, never a success threshold.
    """
    import torch
    c=score.components
    def mean(name):
        values=c[name].double()
        return float(torch.where(valid & torch.isfinite(values),values,torch.zeros_like(values)).mean())
    result=dict(raw_return=float(returns.double().mean()),validity=float(valid.double().mean()),
        whip_quality=mean('mechanics_credit'),
        containment=-float(workspace['workspace_mean_square_excess_m2'].double().mean().sqrt()))
    if stage>=2:result['hit_rate']=mean('geometric_hit_success')
    if stage==2:result['delivery_quality']=mean('target_proximity')
    if stage==3:result['precision_quality']=mean('crossing_center_quality')
    if stage==4:
        result.update(joint_rate=mean('whip_hover_and_cable_success'),drone_hover_rate=mean('recovery_success'),
            cable_settle_rate=mean('cable_settling_success'),recovery_quality=mean('recovery_quality'),
            cable_quality=mean('cable_window_quality'))
    return result


class PlateauDetector:
    def __init__(self,settings):
        validate_plateau(settings)
        self.configuration=deepcopy(settings['plateau_detection'])
        self.stage=settings['training_stage'];self.categories=metric_categories(self.stage)
        self.reset()

    def reset(self):
        self.target=None;self.history=[];self.streak=0;self.last_check=0;self.last_round=0

    def observe(self,collection,metrics,*,target=None,final_target=True,entropy_ready=True):
        c=self.configuration
        if type(collection) is not int or collection<=self.last_round:raise ValueError('Plateau requires distinct increasing completed collections')
        if type(final_target) is not bool or type(entropy_ready) is not bool:raise ValueError('Plateau eligibility must be boolean')
        if self.target!=target or (self.last_round and collection!=self.last_round+1):
            self.reset();self.target=target
        self.last_round=collection
        out=dict(collection=collection,mode=c['mode'],minimum_collections=c['minimum_collections'],
            flagged=False,stop=False,streak=self.streak,measurement='stochastic complete ALL-lane cohort; not independent policy trials')
        if set(metrics)!=set(self.categories) or any(not isinstance(v,(int,float)) or isinstance(v,bool) or not math.isfinite(v) for v in metrics.values()):
            self.history=[];self.streak=0
            return dict(out,reason='missing_or_nonfinite_metrics',streak=0)
        self.history.append(dict(collection=collection,metrics=deepcopy(metrics),entropy_ready=entropy_ready))
        n=c['window_collections'];self.history=self.history[-2*n:]
        reason=None
        if not final_target:reason='target_curriculum_not_final'
        elif collection<c['minimum_collections']:reason='minimum_training_not_reached'
        elif len(self.history)<2*n:reason='insufficient_same_target_history'
        elif c['require_entropy_floor'] and not all(row['entropy_ready'] for row in self.history):reason='entropy_still_annealing_in_comparison_windows'
        if reason:
            self.streak=0
            return dict(out,reason=reason,streak=0)
        if collection-self.last_check<c['check_interval_collections']:
            return dict(out,reason='waiting_for_next_check')
        self.last_check=collection
        old=self.history[:n];new=self.history[n:]
        average=lambda rows,key:sum(row['metrics'][key] for row in rows)/n
        gains={key:average(new,key)-average(old,key) for key in self.categories}
        improvements=[key for key,gain in gains.items() if gain>=c['minimum_gain'][self.categories[key]]]
        self.streak=0 if improvements else self.streak+1
        flagged=self.streak>=c['patience_checks']
        return dict(out,reason='meaningful_improvement' if improvements else 'no_meaningful_improvement',
            gains=gains,improving_metrics=improvements,streak=self.streak,flagged=flagged,
            stop=flagged and c['mode']=='stop',interpretation='plateau or regression, NOT convergence or task success')

    def state_dict(self):
        return dict(schema='stage_plateau_v1',configuration=deepcopy(self.configuration),stage=self.stage,
            target=self.target,history=deepcopy(self.history),streak=self.streak,last_check=self.last_check,last_round=self.last_round)

    def load_state_dict(self,saved):
        if saved.get('schema')!='stage_plateau_v1' or saved.get('configuration')!=self.configuration or saved.get('stage')!=self.stage:
            raise ValueError('Plateau checkpoint configuration/schema mismatch')
        for key in ('streak','last_check','last_round'):
            if type(saved.get(key)) is not int or saved[key]<0:raise ValueError('Invalid plateau saved counter')
        if saved['last_check']>saved['last_round']:raise ValueError('Invalid plateau check clock')
        history=saved.get('history')
        if not isinstance(history,list) or len(history)>2*self.configuration['window_collections']:raise ValueError('Invalid plateau history')
        if saved.get('target') is not None and (type(saved['target']) is not int or saved['target']<0):raise ValueError('Invalid plateau target state')
        previous=0
        for row in history:
            if (set(row)!= {'collection','metrics','entropy_ready'} or type(row['collection']) is not int
                    or not previous<row['collection']<=saved['last_round'] or type(row['entropy_ready']) is not bool
                    or set(row['metrics'])!=set(self.categories)
                    or any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) for v in row['metrics'].values())):
                raise ValueError('Invalid plateau saved observation')
            previous=row['collection']
        if history and previous!=saved['last_round']:raise ValueError('Plateau history clock mismatch')
        for key in ('target','history','streak','last_check','last_round'):setattr(self,key,deepcopy(saved[key]))
