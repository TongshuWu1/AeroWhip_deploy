"""Fixed-state evaluation contracts, not evidence of robust learned whipping."""
from dataclasses import replace
import unittest
from unittest.mock import patch
import torch

from training.configuration import FROZEN_PRESETS, load
from tasks.whip.sac_environment import Stage1Environment
from tasks.whip.deterministic_evaluation import DeterministicEvaluationEnvironment,NativeNumericalFailure
from simulation.tensor_model import ControlledState,controlled_values


class DeterministicEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.settings,cls.snapshot,cls.cfg,cls.scene=load(FROZEN_PRESETS[1])
        cls.settings=dict(cls.settings,device='cpu',evaluation_environments=4)

    def env(self):
        return DeterministicEvaluationEnvironment(self.settings,self.snapshot,self.cfg,self.scene)

    def test_policy_width_layout_and_no_future_observations(self):
        env=self.env()
        def callback(frames,index):
            self.assertEqual(frames.shape,(4,451,99))
            self.assertTrue(frames.is_contiguous())
            self.assertTrue(torch.equal(frames,env.frames.repeat(4,1,1)))
            self.assertTrue(torch.equal(index,env.indices().repeat(4)))
            return torch.ones(4,3)*.2
        self.assertTrue(torch.equal(env.policy_action(callback),torch.ones(1,3)*.2))
        self.assertEqual(env.batch,1)
        with self.assertRaises(ValueError): env.policy_action(lambda h,i:torch.zeros(1,3))

    def test_scalar_tensor_step_parity_and_bdf_history(self):
        native=self.env()
        tensor=Stage1Environment(dict(self.settings,parallel_environments=1),self.snapshot,self.cfg,self.scene,capture=False)
        command=torch.tensor([[.2,-.1,.1,native.hold[3]]],dtype=torch.float64)
        for _ in range(3):
            native.state=native._advance(command)
            tensor.state=tensor._advance(command)
            for x,y in zip(controlled_values(native.state),controlled_values(tensor.state)):
                torch.testing.assert_close(x,y,rtol=0,atol=1e-9)
        self.assertGreater(float(native.state.previous_step_s[0,0]),0)
        # Restore every state field, including history and motors, into another
        # native environment. No model-owned invisible history is relied on.
        other=self.env(); other.state=ControlledState(*(v.clone() for v in controlled_values(native.state)))
        for x,y in zip(controlled_values(native._advance(command)),controlled_values(other._advance(command))):
            torch.testing.assert_close(x,y,rtol=0,atol=0)

    def test_numerical_failure_retains_only_valid_prefix(self):
        env=self.env()
        with patch.object(env,'native_stepper',side_effect=NativeNumericalFailure('test convergence failure')):
            _,_,done=env.step(torch.zeros(1,3))
        self.assertTrue(done)
        self.assertTrue(bool(env.failed[0]))
        self.assertFalse(bool(env.numerical_valid[:,1:].any()))
        self.assertEqual(env.numerical_failure,'test convergence failure')
        self.assertIsNone(env.replay_arrays())
        returns,valid,_,_=env.finish()
        self.assertFalse(bool(valid[0]))
        self.assertEqual(float(returns[0]),self.cfg.returns.failure_return)
        env.reset(); self.assertIsNone(env.numerical_failure)
        self.assertTrue(bool(env.active.all()))

    def test_physical_failure_gate_is_not_bypassed(self):
        env=self.env()
        current=env.state
        rotation=current.rotation.clone()
        rotation[0]=torch.tensor([[1.,0.,0.],[0.,0.,-1.],[0.,1.,0.]],dtype=torch.float64)
        unsafe=replace(current,rotation=rotation)
        with patch.object(env,'native_stepper',return_value=unsafe):
            _,_,done=env.step(torch.zeros(1,3))
        self.assertTrue(done); self.assertTrue(bool(env.failed[0]))
        # The finite envelope-violating endpoint is retained; the subsequent
        # frozen suffix is invalid, just as in the collection adapter.
        self.assertTrue(bool(env.numerical_valid[0,:2].all()))
        self.assertFalse(bool(env.numerical_valid[0,2:].any()))
        _,valid,_,_=env.finish(); self.assertFalse(bool(valid[0]))

    def test_backend_errors_are_not_mislabeled_as_physical_failure(self):
        env=self.env()
        with patch.object(env,'native_stepper',side_effect=RuntimeError('backend failure')):
            with self.assertRaisesRegex(RuntimeError,'backend failure'):
                env.step(torch.zeros(1,3))


if __name__=='__main__': unittest.main()
