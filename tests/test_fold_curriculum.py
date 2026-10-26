"""Approved formation guide, alignment, accounting and handoff constraints."""
from pathlib import Path
from copy import deepcopy
import json,unittest
import numpy as np
import torch

from test_fold_release import analytic_fold
from evaluation.whip_measurements import extract_measurements
from evaluation.whip_output_decomposition import decompose
from evaluation.whip_measurement_examples import kinematic_controls
from tasks.whip.fold_formation import FoldFormationScorer,aligned_release_events,reversal_angles
from tasks.whip.sac_environment import Stage1Environment
from training.configuration import load,collection_limit,validate
from training.artifacts import validate_fold_handoff
from training.worker import record_simple_collection
from training.ppo import PPO
from helpers import project_temporary

PRESET='configs/learning/whip_stage1_ppo_fold_formation_ar1_v1.json'


class FoldCurriculumTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.settings,cls.snapshot,cls.cfg,cls.scene=load(PRESET)
        cls.rest=np.array(cls.snapshot['model']['cable']['marker_interval_lengths_m'])

    def env(self,arrays):
        env=Stage1Environment(dict(self.settings,device='cpu',parallel_environments=1),self.snapshot,self.cfg,self.scene,capture=False)
        env.history['cable_positions'][0].copy_(torch.tensor(arrays['cable_positions']))
        env.history['cable_velocities'][0].copy_(torch.tensor(arrays['cable_velocities']))
        env.history['drone_positions'][:]=env.origin
        env.numerical_valid[:]=True;env.failed[:]=False;env.active[:]=False
        env.decision=env.decisions;env.lengths[:]=env.decisions
        return env

    def aligned(self,a):
        m=extract_measurements(a,self.rest,[1,0,0]);d=decompose(a['cable_positions'],a['cable_velocities'],self.rest)
        return aligned_release_events(a,m,d,[1,0,0],complete=True,failed=False)

    def test_straight_has_zero_and_shape_guide_is_not_release_success(self):
        p=np.zeros((451,len(self.rest)+1,3));p[:,:,2]=2-np.r_[0,np.cumsum(self.rest)]
        a=dict(time_s=np.arange(451)*.01,cable_positions=p,cable_velocities=np.zeros_like(p))
        env=self.env(a);ret,valid,score,_=env.finish()
        self.assertTrue(bool(valid[0]));self.assertEqual(float(ret[0]),0.)
        self.assertEqual(float(score.components['root_fold_formed'][0]),0.)
        curved=kinematic_controls(self.rest)[0]['arrays'];env=self.env(curved)
        ret,valid,score,_=env.finish()
        self.assertGreater(float(ret[0]),0.)
        self.assertEqual(float(score.components['fold_release_score'][0]),0.)
        self.assertFalse(env.scorer.event_records[0]['full_stage1_success'])

    def test_single_frame_does_not_satisfy_persistent_fold(self):
        a=analytic_fold(self.rest);angle=reversal_angles(a['cable_positions'])
        root=np.r_[0,np.cumsum(self.rest)][1:-1]/sum(self.rest)<=.35
        k=int(np.argmax(angle[:,root].max(1)))
        held=deepcopy(a);held['cable_positions'][:]=a['cable_positions'][0]
        held['cable_positions'][100]=a['cable_positions'][k];held['cable_velocities'][:]=0
        env=self.env(held);_,_,score,_=env.finish()
        self.assertGreater(float(score.score[0]),np.pi/2)
        self.assertEqual(float(score.components['root_fold_formed'][0]),0.)
        held['cable_positions'][101]=a['cable_positions'][k]
        env=self.env(held);_,_,score,_=env.finish()
        self.assertEqual(float(score.components['root_fold_formed'][0]),1.)

    def test_lateral_and_backward_output_zero_and_mixed_output_downweighted(self):
        a=analytic_fold(self.rest);straight=self.aligned(a)
        self.assertGreater(straight['best_event_score'],0.)
        for R in (np.array([[0,-1,0],[1,0,0],[0,0,1]]),np.diag([-1.,-1.,1.])):
            b=dict(a,cable_positions=a['cable_positions']@R.T,cable_velocities=a['cable_velocities']@R.T)
            self.assertEqual(self.aligned(b)['best_event_score'],0.)
        angle=np.pi/4;R=np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]])
        b=dict(a,cable_positions=a['cable_positions']@R.T,cable_velocities=a['cable_velocities']@R.T)
        mixed=self.aligned(b)
        self.assertLess(mixed['best_event_score'],straight['best_event_score'])
        for r in (straight,mixed):
            e=r['events'][r['selected_event']]
            self.assertAlmostEqual(e['score'],e['unaligned_fold_release_score']*e['direction_alignment'])
            self.assertAlmostEqual(float(r['causal_best'][-1]),r['best_event_score'])

    def test_dense_angle_accounting_movement_and_failure(self):
        a=analytic_fold(self.rest);env=self.env(a)
        env.history['drone_positions'][0,:,1]+=torch.arange(451,dtype=torch.float64)*.001
        ret,valid,score,_=env.finish()
        expected=float(reversal_angles(a['cable_positions'])[:,:3].max())
        self.assertAlmostEqual(float(score.score[0]),expected,places=10)
        torch.testing.assert_close(env.transition_rewards.sum(1),ret)
        self.assertAlmostEqual(float(ret[0]),expected-.25*.45/sum(self.rest),places=6)
        with project_temporary() as directory:
            directory=Path(directory)
            record_simple_collection(directory,env,ret,valid,score,1)
            record=json.loads((directory/'collection-00001-events.json').read_text())['lanes'][0]
            self.assertIn('formation_only_episode_return',record)
            self.assertNotIn('shape_only_episode_return',record)
            self.assertEqual(record['episode_return'],float(ret[0]))
        env.failed[:]=True;ret,valid,score,_=env.finish()
        self.assertEqual(float(ret[0]),-10.);self.assertFalse(bool(valid[0]))
        torch.testing.assert_close(env.transition_rewards.sum(1),ret)

    def test_review_cap_does_not_compress_entropy_clock(self):
        self.assertEqual(collection_limit(self.settings),30)
        self.assertEqual(self.settings['training_episodes'],15360)
        env=self.env(analytic_fold(self.rest))
        agent=PPO(self.settings,env.actor_columns,env.frame_size,env.device,env.action_size,
                  privileged_size=env.critic_state_size,encoder_options=env.policy_encoder_options)
        self.assertEqual(agent.entropy_episode_budget,15360);self.assertEqual(agent.entropy_episode_offset,0)
        self.assertEqual(agent.entropy_weight,.02)
        agent.set_completed_episodes(3840);coefficient=agent.entropy_weight
        release=load('configs/learning/whip_stage1_ppo_fold_release_ar1_v1.json')[0]
        target=PPO(release,env.actor_columns,env.frame_size,env.device,env.action_size,
                   privileged_size=env.critic_state_size,encoder_options=env.policy_encoder_options)
        target.initialize_actor(agent.state_dict())
        self.assertEqual(target.entropy_episode_budget,15360);self.assertEqual(target.entropy_episode_offset,3840)
        self.assertEqual(target.entropy_weight,coefficient)
        self.assertEqual(len(target.actor_optimizer.state),0);self.assertEqual(len(target.value_optimizer.state),0)

    def test_curriculum_validation_and_exact_review_handoff(self):
        bad=deepcopy(self.settings);bad['fold_curriculum']['warmup_collections']=31
        with self.assertRaises(ValueError):validate(bad)
        release,snapshot,_,_=load('configs/learning/whip_stage1_ppo_fold_release_ar1_v1.json')
        saved=dict(snapshot=self.snapshot,episodes=3840,round=30,agent=dict(entropy_episode_budget=15360))
        with project_temporary() as directory:
            directory=Path(directory)
            path=directory/'review.json';snapshot['settings']['fold_curriculum']['motion_review']=str(path)
            source=directory/'latest.pt'
            (directory/'summary.json').write_text(json.dumps(dict(phase='stopped',episodes=3840,round=30)))
            review=dict(schema='stage1_fold_formation_review_v1',approved=True,numeric_gate_passed=True,
                motion_review_passed=True,checkpoint_sha256='a'*64,formation_episodes=3840,evidence=['retained review evidence'])
            path.write_text(json.dumps(review));validate_fold_handoff(saved,snapshot,'a'*64,source)
            with self.assertRaises(ValueError):validate_fold_handoff(saved,snapshot,'b'*64,source)
            earlier=dict(saved,episodes=3584,round=28)
            with self.assertRaises(ValueError):validate_fold_handoff(earlier,snapshot,'a'*64,source)
            review['motion_review_passed']=False;path.write_text(json.dumps(review))
            with self.assertRaises(ValueError):validate_fold_handoff(saved,snapshot,'a'*64,source)
