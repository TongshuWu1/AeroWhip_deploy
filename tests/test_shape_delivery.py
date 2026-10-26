"""Same-event Stage2 counterexamples and transfer/accounting, not physics validation."""
from copy import deepcopy
from pathlib import Path
import json
import unittest
import numpy as np
import torch

from helpers import project_temporary, wave
from training.configuration import load
from training.artifacts import identity, validate_actor_source, execution_hashes
from training.ppo import PPO
from training.worker import record_simple_collection
from tasks.whip.shape_delivery import delivery_events
from tasks.whip.shape_release import ShapeReleaseScorer
from tasks.whip.sac_environment import Stage1Environment

PRESET='configs/learning/whip_stage2_ppo_shape_delivery_path_ar1_v1.json'


def event(id=0,arm=1,end=3,q=2,score=2.):
    return dict(id=id,baseline_index=0,arming_index=arm,end_index=end,q_shape_index=q,
                score=score,finalized=True,provisional_eligible=True)


class DeliveryEventTests(unittest.TestCase):
    def evaluate(self,events,*,positions=None,velocities=None,count=5,failed=False):
        time=np.arange(5)*.01
        p=np.array([[0.,0,2],[.5,0,2],[1.1,0,2],[1.5,0,2],[2,0,2]]) if positions is None else positions
        v=np.tile([4.,0,0],(5,1)) if velocities is None else velocities
        core=dict(events=events,selected_event=0 if events else None,best_event_score=max([e['score'] for e in events]+[0]),complete=count==5 and not failed,failed=failed)
        return delivery_events(core,time[:count],p[:count],v[:count],center=[1.1,0,2],width=1,height=.6,sigma=.7)

    def test_same_q_and_no_independent_closest_approach(self):
        r=self.evaluate([event()]);self.assertEqual(r['best_event_score'],4)
        self.assertEqual(self.evaluate([])['best_event_score'],0)
        self.assertEqual(self.evaluate([event(score=0)])['best_event_score'],0)
        r=self.evaluate([event(q=1)])
        self.assertAlmostEqual(r['best_event_score'],2*(1+np.exp(-.5*(.6/.7)**2)))
        np.testing.assert_array_equal(r['causal_best'][:3],0)
        self.assertEqual(r['causal_best'][3],r['best_event_score'])

    def test_maximum_earliest_tie_and_no_core_mutation(self):
        events=[event(id=9),event(id=2)];before=deepcopy(events);r=self.evaluate(events)
        self.assertEqual(r['selected_event'],0);self.assertEqual(r['best_event_score'],4)
        self.assertEqual(events,before)
        # Finalization order can differ from event-list order; causal max remains exact.
        r=self.evaluate([event(end=4,score=3),event(end=2,score=2)])
        self.assertEqual(r['causal_best'][2],4);self.assertEqual(r['causal_best'][4],6)

    def test_crossing_endpoints_direction_and_speed_diagnostics(self):
        for kw in ({},{'arm':2},{'end':2}):
            self.assertEqual(len(self.evaluate([event(**kw)])['events'][0]['associated_crossings']),1)
        for kw in ({'arm':3,'q':3},{'end':1,'q':1}):
            self.assertEqual(self.evaluate([event(**kw)])['events'][0]['associated_crossings'],[])
        p=np.array([[2.,0,2],[1.5,0,2],[1.1,0,2],[.5,0,2],[0,0,2]])
        self.assertEqual(self.evaluate([event()],positions=p)['global_forward_crossings'],[])
        self.assertEqual(self.evaluate([event()],velocities=np.tile([-4.,0,0],(5,1)))['global_forward_crossings'],[])
        r=self.evaluate([event()],velocities=np.tile([1.,0,0],(5,1)))
        crossing=r['events'][0]['associated_crossings'][0]
        self.assertTrue(crossing['geometric_hit']);self.assertFalse(crossing['strict_hit'])

    def test_interpolated_boundary_and_unrelated_event(self):
        p=np.array([[0.,.5,2.3],[.5,.5,2.3],[1.7,.5,2.3],[2,.5,2.3],[3,.5,2.3]])
        r=self.evaluate([event()],positions=p);x=r['events'][0]['associated_crossings'][0]
        self.assertAlmostEqual(x['fraction'],.5);self.assertAlmostEqual(x['time_s'],.015)
        self.assertTrue(x['geometric_hit'])
        r=self.evaluate([event(score=.1),event(id=1,arm=3,q=3,end=4,score=3)])
        self.assertEqual(r['selected_event'],1);self.assertEqual(r['events'][1]['associated_crossings'],[])
        self.assertTrue(r['events'][0]['associated_crossings'][0]['geometric_hit'])

    def test_failed_credit_and_observed_prefix(self):
        r=self.evaluate([event()],failed=True);self.assertEqual(r['episode_return'],-10)
        self.assertEqual(r['causal_best'][-1],4)
        unfinished=event();unfinished['finalized']=False
        r=self.evaluate([unfinished],count=2,failed=True)
        self.assertEqual(r['best_event_score'],0);self.assertEqual(r['global_forward_crossings'],[])
        with self.assertRaises(ValueError):self.evaluate([event()],count=2)


class DeliveryIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.settings,cls.snapshot,cls.cfg,cls.scene=load(PRESET)
        cls.rest=torch.tensor(cls.snapshot['model']['cable']['marker_interval_lengths_m'],dtype=torch.float64)
        cls.time=torch.arange(451,dtype=torch.float64)*.01
        cls.p,cls.v=wave(cls.rest,cls.time)

    def environment(self):
        env=Stage1Environment(dict(self.settings,device='cpu',parallel_environments=1),self.snapshot,self.cfg,self.scene,capture=False)
        env.history['cable_positions'].copy_(self.p);env.history['cable_velocities'].copy_(self.v)
        env.history['drone_positions'][:]=env.origin;env.history['drone_positions'][0,:,0]=torch.arange(451)*.005
        env.numerical_valid[:]=True;env.failed[:]=False;env.active[:]=False;env.decision=env.decisions;env.lengths[:]=env.decisions
        return env

    def test_frozen_core_preserved_dense_cost_failure_and_logging(self):
        env=self.environment();r,v,score,_=env.finish()
        base=ShapeReleaseScorer(self.rest,self.time,self.cfg,initial_position=env.origin)(self.p,self.v,env.numerical_valid,failed=env.failed)
        torch.testing.assert_close(score.components['shape_release_score'],base.score,rtol=0,atol=0)
        torch.testing.assert_close(env.transition_rewards.sum(1),r,rtol=1e-6,atol=1e-6)
        torch.testing.assert_close(r,(score.score-score.components['movement_cost']).float())
        record=env.scorer.event_records[0];self.assertEqual(record['schema'],'directed_shape_delivery_v1')
        for e in record['events']:
            if e['score']>0:self.assertAlmostEqual(e['delivery_utility'],e['score']*(1+e['target_guidance']))
        with project_temporary() as folder:
            record_simple_collection(folder,env,r,v,score,1)
            saved=json.loads((Path(folder)/'collection-00001-events.json').read_text())
            self.assertIn('event_only_episode_return',saved['lanes'][0]);self.assertEqual(saved['lanes'][0]['episode_return'],float(r[0]))
            self.assertNotIn('movement_cost',record)
        env.failed[:]=True;env.lengths[:]=150;env.numerical_valid[:,301:]=False
        r,v,score,_=env.finish();self.assertFalse(bool(v[0]));self.assertEqual(float(r[0]),-10)
        self.assertEqual(float(score.components['movement_cost'][0]),0)
        torch.testing.assert_close(env.transition_rewards.sum(1),r,rtol=1e-6,atol=1e-6)
        self.assertEqual(int(env.transition_rewards[0,150:].count_nonzero()),0)

    def test_configuration_and_narrow_transfer(self):
        for key,value in (('training_stage',1),('discount',.99),('workspace_penalty',dict(weight=.1,distance_scale_m=.1))):
            bad=deepcopy(self.settings);bad[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):load(bad)
        env=self.environment();oldsettings,oldsnapshot,_,_=load('configs/learning/whip_stage1_ppo_shape_release_path_ar1_v3.json')
        old=PPO(dict(oldsettings,device='cpu'),env.actor_columns,env.frame_size,'cpu',privileged_size=env.critic_state_size,encoder_options=env.policy_encoder_options)
        old.set_completed_episodes(14336);old.entropy_episode_offset=10752
        oldsnapshot['execution_hashes']=execution_hashes()
        oldsnapshot['identity']=identity(oldsnapshot)
        saved=dict(schema='ppo_stage1_checkpoint_v1',snapshot=oldsnapshot,identity=oldsnapshot['identity'],agent=old.state_dict())
        validate_actor_source(saved,self.snapshot)
        target=PPO(dict(self.settings,device='cpu'),env.actor_columns,env.frame_size,'cpu',privileged_size=env.critic_state_size,encoder_options=env.policy_encoder_options)
        critic=deepcopy(target.value.state_dict());target.initialize_actor(saved['agent'])
        for k,x in old.actor.state_dict().items():torch.testing.assert_close(x,target.actor.state_dict()[k],rtol=0,atol=0)
        for k,x in critic.items():torch.testing.assert_close(x,target.value.state_dict()[k],rtol=0,atol=0)
        self.assertEqual(target.entropy_episode_offset,25088);self.assertEqual(target.entropy_episode_budget,15360);self.assertAlmostEqual(target.entropy_weight,.001)
        self.assertFalse(target.actor_optimizer.state);self.assertFalse(target.value_optimizer.state)
        legacy=load('configs/learning/whip_stage2_ppo_acceleration_v1.json')[1]
        with self.assertRaises(ValueError):validate_actor_source(saved,legacy)
        bounded=deepcopy(saved);bounded['snapshot']['task']['matching']['reward_mode']='directed_discovery_bounded_v1';bounded['snapshot']['identity']=identity(bounded['snapshot']);bounded['identity']=bounded['snapshot']['identity']
        with self.assertRaises(ValueError):validate_actor_source(bounded,self.snapshot)
