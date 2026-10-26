"""State-machine counterexamples with explicit synthetic descriptor sequences.

These isolate event logic; they are not physically simulated whip examples.
"""
import unittest
import math
import numpy as np
from tasks.whip.horizontal_fold import HorizontalFoldAttempt

class FoldEventTests(unittest.TestCase):
    def setUp(self):
        self.a=HorizontalFoldAttempt(np.full(10,.1))
        self.p=np.zeros((11,3));self.p[-1]=[.8,0,0]
        self.v=np.zeros((11,3));self.v[-1]=[1,0,0]
        self.t=0;self.rewards=[]
    def step(self,centre=.3,turn=3.,ident=1,flags=(),outward=True,reach=.8,rows=True):
        row=dict(track_id=ident,centre_m=centre,theta=turn,coherence=1.,joint_indices=[1],unresolved=False,flags=list(flags))
        self.a.geometry.observe=lambda _: (np.array([turn]+[0.]*8),[row] if rows else [])
        self.t+=1;self.p[-1,0]=reach;self.v[-1,0]=1 if outward else -1
        self.rewards.append(self.a.observe(self.t*.01,self.p,self.v))
    def arm_propagate(self):
        for _ in range(3):self.step()
        self.step(centre=.7,turn=2.)
        self.assertTrue(self.a.propagated)
    def test_opening_buffer_selects_earliest_equal_maximum(self):
        self.arm_propagate()
        for _ in range(3):self.step(centre=.8,turn=.5)
        self.assertTrue(self.a.opened);self.assertEqual(self.a.selected['time_s'],.05)
        for _ in range(3):self.step(centre=.8,turn=.5,outward=False,reach=.9)
        self.assertEqual(self.a.terminal,'event_end');self.assertTrue(self.a.qualified)
        self.assertAlmostEqual(sum(self.rewards),5*.8**2-.1*.1,places=12)
        with self.assertRaises(RuntimeError):self.step(centre=.9,turn=.1,reach=1.)
    def test_unconfirmed_opening_credit_is_discarded(self):
        self.arm_propagate();self.step(centre=.8,turn=.5)
        self.assertIsNotNone(self.a.selected)
        self.step(centre=.8,turn=2.)
        self.assertIsNone(self.a.selected);self.assertFalse(self.a.opened)
    def test_two_bad_samples_recover_three_end(self):
        self.arm_propagate()
        for _ in range(3):self.step(centre=.8,turn=.5)
        for _ in range(2):self.step(centre=.8,turn=2.)
        self.assertIsNone(self.a.terminal)
        self.step(centre=.8,turn=.5,reach=.85)
        self.assertEqual(self.a.active['bad'],0)
        # Any combination of refolding / radial reversal contributes to one counter.
        self.step(centre=.8,turn=2.)
        self.step(centre=.8,turn=.5,outward=False)
        self.step(centre=.8,turn=2.)
        self.assertEqual(self.a.terminal,'event_end');self.assertEqual(self.a.selected['H'],.85)
    def test_ambiguous_track_discards_and_rearms_only_after_persistence(self):
        for _ in range(3):self.step()
        self.step(ident=2,flags=['ambiguous_identity'])
        self.assertIsNone(self.a.active)
        for _ in range(2):self.step(ident=2)
        self.assertIsNone(self.a.active)
        self.step(ident=2);self.assertEqual(self.a.active['id'],2)
    def test_failed_opening_with_identity_gap_cannot_rejoin(self):
        self.arm_propagate();self.step(centre=.8,turn=.5,rows=False)
        self.assertTrue(self.a.active['identity_lost'])
        self.step(centre=.8,turn=2.)
        self.assertIsNone(self.a.active);self.assertIsNone(self.a.selected)
    def test_confirmed_output_without_alignment_is_zero_credit(self):
        self.arm_propagate();self.v[-1,2]=3.
        for _ in range(3):self.step(centre=.8,turn=.5)
        for _ in range(3):self.step(centre=.8,turn=.5,outward=False)
        self.assertEqual(self.a.terminal,'event_end');self.assertIsNone(self.a.selected)
        self.assertFalse(self.a.qualified);self.assertAlmostEqual(sum(self.rewards),-.01,places=12)
    def test_event_completion_at_deadline_precedes_timeout(self):
        self.arm_propagate()
        for _ in range(3):self.step(centre=.8,turn=.5)
        while self.t<697:self.step(centre=.8,turn=.5)
        for _ in range(3):self.step(centre=.8,turn=.5,outward=False)
        self.assertEqual(self.t,700);self.assertEqual(self.a.terminal,'event_end')
        self.assertAlmostEqual(sum(self.rewards),5*.8**2-.7,places=12)
    def test_failure_cancels_selected_output(self):
        self.arm_propagate()
        for _ in range(3):self.step(centre=.8,turn=.5)
        self.rewards.append(self.a.fail(.071,'workspace_failure'))
        self.assertFalse(self.a.qualified);self.assertAlmostEqual(sum(self.rewards),-10.0071,places=12)

if __name__=='__main__':unittest.main()
