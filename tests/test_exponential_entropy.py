"""Approved absolute-episode exponential entropy schedule."""
from copy import deepcopy
import math
from types import SimpleNamespace
import unittest

from training.configuration import load
from training.ppo import PPO, validate_ppo


class ExponentialEntropyTests(unittest.TestCase):
    def settings(self):
        return load()[0]

    def weight(self, episodes, *, budget=12800, lanes=64):
        settings=self.settings();settings['parallel_environments']=lanes
        a=SimpleNamespace(settings=settings,completed_episodes=episodes,entropy_episode_budget=budget)
        return PPO.entropy_weight.fget(a)

    def test_approved_curve_and_fixed_episode_timescale(self):
        for n in (0,1,20,40,60,81,120):
            expected=.001+.019*math.exp(-8*n/200)
            self.assertAlmostEqual(self.weight(n*64),expected,places=15)
            self.assertEqual(self.weight(n*64),self.weight(n*128,budget=25600,lanes=1))

    def test_fast_early_decrease_then_flatten_with_positive_floor(self):
        values=[self.weight(n*64) for n in range(121)]
        drops=[a-b for a,b in zip(values,values[1:])]
        self.assertTrue(all(d>0 for d in drops))
        self.assertTrue(all(a>b for a,b in zip(drops,drops[1:])))
        self.assertGreater(values[-1],.001)
        self.assertEqual(self.weight(10**9),.001)

    def test_clock_units_compatible_with_existing_episode_state(self):
        a=SimpleNamespace(settings=self.settings())
        self.assertEqual(PPO.entropy_clock.fget(a),'completed_episodes_v1')

    def test_invalid_timescale_and_schema_rejected(self):
        for value in (0,-1,True,float('nan')):
            s=self.settings()['ppo'];s['entropy']['decay_exponent']=value
            with self.assertRaises(ValueError):validate_ppo(s)
        for edit in ({'mode':'unknown'},{'hold_fraction':.1}):
            s=self.settings()['ppo'];s['entropy'].update(edit)
            with self.assertRaises(ValueError):validate_ppo(s)


if __name__=='__main__':unittest.main()
