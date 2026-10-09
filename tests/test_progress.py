"""Current entropy and plateau clocks: stopping is not convergence."""
from copy import deepcopy
from types import SimpleNamespace
import unittest
import torch
from training.configuration import FROZEN_PRESETS, load
from training.plateau import PlateauDetector, collection_metrics, metric_categories
from training.ppo import PPO


class ProgressTests(unittest.TestCase):
    def settings(self, stage=1):
        return load(FROZEN_PRESETS[stage])[0]

    def metrics(self, stage=1):
        return {k: .5 for k in metric_categories(stage)}

    def test_never_stop_before40_completed_collections(self):
        for stage in (1,2,3,4):
            d=PlateauDetector(self.settings(stage))
            for n in range(1,40):
                result=d.observe(n,self.metrics(stage),entropy_ready=True)
                self.assertFalse(result['stop']);self.assertFalse(result['flagged'])
            self.assertFalse(d.observe(40,self.metrics(stage))['stop'])
            self.assertFalse(d.observe(41,self.metrics(stage))['stop'])
            self.assertTrue(d.observe(42,self.metrics(stage))['stop'])

    def test_improvement_in_any_outcome_resets_patience(self):
        for key in metric_categories(4):
            d=PlateauDetector(self.settings(4))
            for n in range(1,41):d.observe(n,self.metrics(4))
            metrics=self.metrics(4);metrics[key]=2.
            result=d.observe(41,metrics)
            self.assertIn(key,result['improving_metrics']);self.assertEqual(result['streak'],0)
            self.assertFalse(result['stop'])

    def test_target_curriculum_not_final_and_change_reset(self):
        d=PlateauDetector(self.settings(3))
        for n in range(1,61):self.assertFalse(d.observe(n,self.metrics(3),target=0,final_target=False)['stop'])
        first=d.observe(61,self.metrics(3),target=3,final_target=True)
        self.assertEqual(first['reason'],'insufficient_same_target_history')
        for n in range(62,76):self.assertFalse(d.observe(n,self.metrics(3),target=3)['stop'])
        self.assertFalse(d.observe(76,self.metrics(3),target=3)['stop'])
        self.assertFalse(d.observe(77,self.metrics(3),target=3)['stop'])
        self.assertTrue(d.observe(78,self.metrics(3),target=3)['stop'])

    def test_warn_only_does_not_stop(self):
        s=self.settings();s['plateau_detection']['mode']='warn';d=PlateauDetector(s)
        for n in range(1,60):
            result=d.observe(n,self.metrics());self.assertFalse(result['stop'])
        self.assertTrue(result['flagged'])

    def test_missing_nonfinite_and_skipped_collection_reset(self):
        d=PlateauDetector(self.settings())
        for n in range(1,41):d.observe(n,self.metrics())
        bad=self.metrics();bad['raw_return']=float('nan')
        result=d.observe(41,bad);self.assertFalse(result['stop']);self.assertEqual(d.history,[])
        d.observe(42,self.metrics())
        result=d.observe(44,self.metrics())
        self.assertEqual(result['reason'],'insufficient_same_target_history')
        self.assertEqual(len(d.history),1)
        with self.assertRaises(ValueError):d.observe(44,self.metrics())

    def test_resume_retains_history_and_patience_exact(self):
        s=self.settings();a=PlateauDetector(s)
        for n in range(1,41):a.observe(n,self.metrics())
        b=PlateauDetector(s);b.load_state_dict(a.state_dict())
        for n in range(41,51):
            self.assertEqual(a.observe(n,self.metrics()),b.observe(n,self.metrics()))
        self.assertEqual(a.state_dict(),b.state_dict())
        bad=a.state_dict();bad['configuration']['minimum_collections']=41
        with self.assertRaises(ValueError):b.load_state_dict(bad)
        bad=a.state_dict();bad['history'][-1]['metrics']['raw_return']=float('nan')
        with self.assertRaises(ValueError):b.load_state_dict(bad)

    def test_all_lane_failed_positive_credit_zero_and_raw_returns_kept(self):
        components={k:torch.tensor([.8,.9]) for k in ('mechanics_credit','geometric_hit_success','target_proximity',
            'crossing_center_quality','whip_hover_and_cable_success','recovery_success','cable_settling_success',
            'recovery_quality','cable_window_quality')}
        valid=torch.tensor([True,False]);returns=torch.tensor([1.,-2.])
        workspace={'workspace_mean_square_excess_m2':torch.tensor([.01,.09])}
        for stage in (1,2,3,4):
            result=collection_metrics(stage,returns,valid,SimpleNamespace(components=components),workspace)
            self.assertEqual(set(result),set(metric_categories(stage)))
            self.assertEqual(result['raw_return'],-.5);self.assertEqual(result['validity'],.5)
            self.assertAlmostEqual(result['whip_quality'],.4)
            self.assertAlmostEqual(result['containment'],-(.05**.5),places=7)

    def test_fresh_budget_entropy_scales_to_collection_count(self):
        for stage in (1,2,3,4):
            s40=self.settings(stage);s40['training_episodes']=5120
            s80=self.settings(stage);s80['training_episodes']=10240
            s100=self.settings(stage);s100['training_episodes']=12800
            options=(dict(frame_size=101,privileged_size=205,encoder_options=dict(history_samples=451,phase_columns=[96,97],goal_columns=[98,99,100]))
                if stage==4 else dict(frame_size=9,privileged_size=5,encoder_options=dict(goal_columns=[6,7,8]))
                )
            a=PPO(s40,[0,1,2],device='cpu',**options)
            c=PPO(s80,[0,1,2],device='cpu',**options)
            b=PPO(s100,[0,1,2],device='cpu',**options)
            for fraction in (0,.1,.2,.5,.8,1):
                a.set_completed_episodes(round(5120*fraction));b.set_completed_episodes(round(12800*fraction))
                c.set_completed_episodes(round(10240*fraction))
                self.assertAlmostEqual(a.entropy_weight,b.entropy_weight,places=13)
                self.assertAlmostEqual(a.entropy_weight,c.entropy_weight,places=13)
        s=self.settings(3)
        self.assertEqual((s['ppo']['entropy']['hold_fraction'],s['ppo']['entropy']['anneal_end_fraction']),(.1,.8))
