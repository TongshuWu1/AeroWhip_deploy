"""Recorded-negative and kinematic/software tests; no physical-whip certification."""
from pathlib import Path
from copy import deepcopy
import hashlib,json,unittest
import numpy as np
from evaluation.whip_candidate_review import evaluate, distal_motion_fraction, event_checks
from evaluation.whip_measurement_examples import _chain,kinematic_controls

FIXTURES=Path(__file__).resolve().parents[1]/'evaluation/fixtures'


def outward_packet(rest, *, amplitude=-.6, width_fraction=.12, speed_lengths_per_s=2., reverse=False):
    t=np.arange(451)*.01;L=rest.sum();s=np.cumsum(rest)[:-1]
    speed=speed_lengths_per_s*L
    center=(1.2*L-speed*t if reverse else -.2*L+speed*t)
    rate=-speed if reverse else speed;width=width_fraction*L
    turns=amplitude*np.exp(-.5*((s[None]-center[:,None])/width)**2)
    changes=turns*(s[None]-center[:,None])*rate/width**2
    p,v=_chain(rest,np.column_stack((np.zeros(len(t)),np.cumsum(turns,1))),np.column_stack((np.zeros(len(t)),np.cumsum(changes,1))))
    return dict(time_s=t,cable_positions=p+[0,0,2],cable_velocities=v)


class CandidateReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        manifest=json.loads((FIXTURES/'stage1_side_swing_v1.json').read_text())
        source=FIXTURES/manifest['arrays'];assert hashlib.sha256(source.read_bytes()).hexdigest()==manifest['sha256']
        with np.load(source) as z:cls.recorded={k:z[k] for k in z.files}
        cls.rest=cls.recorded['rest_lengths_m'];cls.direction=cls.recorded['direction_world']

    def test_real_side_swing_is_not_accepted_by_candidate(self):
        r=evaluate(self.recorded,self.rest,self.direction)
        self.assertGreater(r['legacy_proxy_score'],6.)
        self.assertEqual(r['candidate_output'],0.)
        self.assertFalse(r['candidate_support'])
        e=r['events'][r['legacy_selected_event']]
        self.assertIn('boundary_censored_release_peak',e['reasons'])
        self.assertIn('no_increase_in_distal_shape_motion_fraction',e['reasons'])
        self.assertEqual(r['physical_whip_status'],'unvalidated')

    def test_constructed_packet_preserves_positive_kinematic_signal(self):
        r=evaluate(outward_packet(self.rest),self.rest,self.direction)
        self.assertGreater(r['candidate_output'],0.)
        self.assertTrue(r['candidate_support'])
        self.assertEqual(r['goal_acceptance'],'unresolved')
        self.assertEqual(r['physical_whip_status'],'unvalidated')

    def test_rejecting_case_does_not_use_y_excursion_or_filename(self):
        a=outward_packet(self.rest);t=a['time_s'];shift=np.column_stack((0*t,10*t,0*t));boost=np.array([0.,10.,0.])
        changed=dict(a,cable_positions=a['cable_positions']+shift[:,None],cable_velocities=a['cable_velocities']+boost)
        r=evaluate(changed,self.rest,self.direction)
        self.assertTrue(r['candidate_support'])
        self.assertAlmostEqual(r['candidate_output'],evaluate(a,self.rest,self.direction)['candidate_output'],places=11)

    def test_rigid_rotation_translation_and_backward_packet_rejected(self):
        cases=kinematic_controls(self.rest)
        for case in cases[:2]:
            with self.subTest(case=case['name']):self.assertFalse(evaluate(case['arrays'],self.rest,self.direction)['candidate_support'])
        self.assertFalse(evaluate(outward_packet(self.rest,reverse=True),self.rest,self.direction)['candidate_support'])

    def test_origin_translation_and_rotation_covariance(self):
        angle=.7;R=np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]])
        for a in (self.recorded,outward_packet(self.rest)):
            before=evaluate(a,self.rest,self.direction)
            changed=dict(a,cable_positions=a['cable_positions']@R.T+[3,-4,7],cable_velocities=a['cable_velocities']@R.T)
            after=evaluate(changed,self.rest,self.direction@R.T)
            self.assertEqual(after['candidate_support'],before['candidate_support'])
            self.assertAlmostEqual(after['candidate_output'],before['candidate_output'],places=11)

    def test_baseline_and_boundary_checks_are_independent(self):
        e=dict(finalized=True,provisional_eligible=True,score=2.,baseline_index=0,arming_index=1,q_shape_index=2,end_index=3)
        r=event_checks(e,np.array([.2,.3,.8,.4]));self.assertTrue(r['candidate_support'])
        r=event_checks(dict(e,q_shape_index=1),np.array([.2,.8,.3,.4]));self.assertEqual(r['reasons'],['boundary_censored_release_peak'])
        r=event_checks(e,np.array([.8,.3,.2,.4]));self.assertEqual(r['reasons'],['no_increase_in_distal_shape_motion_fraction'])
        self.assertIn('boundary_censored_release_peak',event_checks(dict(e,q_shape_index=3),np.array([.2,.3,.4,.8]))['reasons'])

    def test_failure_prefix_and_observation_gap_do_not_pass(self):
        a=outward_packet(self.rest)
        self.assertFalse(evaluate(a,self.rest,self.direction,failed=True)['candidate_support'])
        observed=np.ones(451,bool);observed[40]=False
        r=evaluate(dict(a,state_valid=observed),self.rest,self.direction)
        self.assertEqual(r['observed_samples'],40);self.assertFalse(r['candidate_support'])
        bad=deepcopy(a);bad['cable_positions'][40:]=np.nan
        self.assertFalse(evaluate(bad,self.rest,self.direction)['candidate_support'])

    def test_input_preserved_and_invalid_fraction_data_refused(self):
        a=outward_packet(self.rest);p=a['cable_positions'].copy();evaluate(a,self.rest,self.direction)
        np.testing.assert_array_equal(a['cable_positions'],p)
        with self.assertRaises(ValueError):distal_motion_fraction(np.zeros((2,11,3)),np.zeros(10))
        with self.assertRaises(ValueError):evaluate(a,self.rest,self.direction,complete=1)
