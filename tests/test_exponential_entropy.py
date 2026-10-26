"""Approved absolute-episode exponential schedule and legacy compatibility."""
from copy import deepcopy
import math
from types import SimpleNamespace
import unittest

from training.configuration import load
from training.ppo import PPO, validate_ppo


class ExponentialEntropyTests(unittest.TestCase):
    def settings(self):
        return load('configs/learning/whip_stage1_ppo_unified_fold_ar1_exp_v2.json')[0]

    def weight(self, episodes, *, offset=0, budget=15360, lanes=128):
        settings=self.settings();settings['parallel_environments']=lanes
        a=SimpleNamespace(settings=settings,completed_episodes=episodes,
            entropy_episode_offset=offset,entropy_episode_budget=budget)
        return PPO.entropy_weight.fget(a)

    def test_approved_curve_and_fixed_episode_timescale(self):
        for n in (0,1,20,40,60,81,120):
            expected=.001+.019*math.exp(-n/20)
            self.assertAlmostEqual(self.weight(n*128),expected,places=15)
            self.assertEqual(self.weight(n*128),self.weight(n*128,budget=30720,lanes=1))

    def test_fast_early_decrease_then_flatten_with_positive_floor(self):
        values=[self.weight(n*128) for n in range(121)]
        drops=[a-b for a,b in zip(values,values[1:])]
        self.assertTrue(all(d>0 for d in drops))
        self.assertTrue(all(a>b for a,b in zip(drops,drops[1:])))
        self.assertGreater(values[-1],.001)
        self.assertEqual(self.weight(10**9),.001)

    def test_inherited_offset_is_not_restarted(self):
        self.assertEqual(self.weight(0,offset=10368),self.weight(10368))
        self.assertEqual(self.weight(128,offset=10368),self.weight(10496))

    def test_clock_units_compatible_with_existing_episode_state(self):
        a=SimpleNamespace(settings=self.settings())
        self.assertEqual(PPO.entropy_clock.fget(a),'completed_episodes_v1')

    def test_invalid_timescale_and_schema_rejected(self):
        for value in (0,-1,True,1.5,float('nan')):
            s=self.settings()['ppo'];s['entropy']['decay_episodes']=value
            with self.assertRaises(ValueError):validate_ppo(s)
        for edit in ({'mode':'unknown'},{'hold_fraction':.1}):
            s=self.settings()['ppo'];s['entropy'].update(edit)
            with self.assertRaises(ValueError):validate_ppo(s)

    def test_legacy_linear_unchanged(self):
        settings=load('configs/learning/whip_stage1_ppo_unified_fold_ar1_v1.json')[0]
        a=SimpleNamespace(settings=settings,completed_episodes=0,entropy_episode_offset=0,
            entropy_episode_budget=15360,entropy_clock='completed_episodes_v1')
        for n,expected in ((0,.02),(12,.02),(54,.0105),(96,.001),(120,.001)):
            a.completed_episodes=n*128
            self.assertAlmostEqual(PPO.entropy_weight.fget(a),expected,places=15)

    def test_preset_only_changes_entropy(self):
        old=deepcopy(load('configs/learning/whip_stage1_ppo_unified_fold_ar1_v1.json')[0])
        new=self.settings()
        old['ppo'].pop('entropy');new['ppo'].pop('entropy')
        self.assertEqual(old,new)


if __name__=='__main__':unittest.main()
