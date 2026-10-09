"""Current PPO math, causality, optimizer rollback and restart contracts."""
from copy import deepcopy
import io
import math
import unittest
import torch
from torch import nn
from unittest.mock import patch
from training.ppo import advantages, clipped_loss, gaussian_kl, squashed_log_prob, explained_variance, bounded_actor_step
from helpers import PPOFixture, exact_state


class OptimizerTests(PPOFixture, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_gae_hand_calculation_terminal_and_padding(self):
        r = torch.tensor([[1.,2.,3.],[4.,99.,99.]])
        v = torch.tensor([[.1,.2,.3],[.4,999.,999.]])
        a, target, valid = advantages(r,v,torch.tensor([3,1]),1.,1.)
        torch.testing.assert_close(target,torch.tensor([[6.,5.,3.],[4.,0.,0.]]))
        torch.testing.assert_close(a[0],torch.tensor([5.9,4.8,2.7]))
        self.assertEqual(valid.tolist(),[[True,True,True],[True,False,False]])
        with self.assertRaises(ValueError): advantages(r,v,torch.tensor([2.5,1.]),1.,1.)
        # Full-horizon failure clawback is part of the reward, not a bootstrap.
        _, target, _ = advantages(torch.tensor([[.5,.5,-2.]]),torch.zeros(1,3),torch.tensor([3]),1.,1.)
        torch.testing.assert_close(target,torch.tensor([[-1.,-1.5,-2.]]))

    def test_gae_lambda_matches_reverse_recursion(self):
        r = torch.tensor([[1.,2.,3.]], dtype=torch.float64)
        v = torch.tensor([[.1,.2,.3]], dtype=torch.float64)
        a,_,_ = advantages(r,v,torch.tensor([3]),1.,.95)
        last=3-.3; middle=2+.3-.2+.95*last; first=1+.2-.1+.95*middle
        torch.testing.assert_close(a,torch.tensor([[first,middle,last]],dtype=torch.float64))

    def test_ratio_clipping_both_advantage_signs(self):
        ratio=torch.tensor([1.5,.5,1.5,.5],requires_grad=True)
        adv=torch.tensor([1.,1.,-1.,-1.])
        loss,_=clipped_loss(ratio.log(),torch.zeros(4),adv,.2)
        self.assertAlmostEqual(float(loss.detach()),.15,places=6)
        loss.backward()
        self.assertEqual(float(ratio.grad[0]),0)
        self.assertEqual(float(ratio.grad[3]),0)
        self.assertLess(float(ratio.grad[1]),0)
        self.assertGreater(float(ratio.grad[2]),0)

    def test_squashed_density_saturation_and_entropy_gradient(self):
        raw=torch.tensor([[20.,-20.,0.]],dtype=torch.float64)
        mean=torch.zeros_like(raw); std=torch.zeros_like(raw)
        self.assertTrue(bool(torch.isfinite(squashed_log_prob(raw,mean,std)).all()))
        torch.manual_seed(7)
        std=torch.zeros(100000,1,dtype=torch.float64,requires_grad=True)
        raw=std.exp()*torch.randn_like(std)
        bounded=-squashed_log_prob(raw,torch.zeros_like(raw),std).mean()
        gaussian=(std+.5*math.log(2*math.pi*math.e)).mean()
        self.assertLess(float(bounded.detach()),float(gaussian.detach())-.1)
        self.assertLess(float(bounded.detach()),math.log(2)+.01)
        bounded.backward(); self.assertTrue(bool(torch.isfinite(std.grad).all()))

    def test_actor_causality_and_privileged_value_isolation(self):
        agent=self.agent()
        nn.init.normal_(agent.actor.mean.weight,std=.1)
        h=torch.randn(4,11,6); i=torch.full((4,),3,dtype=torch.long)
        altered=h.clone(); altered[:,4:]=1000
        torch.testing.assert_close(agent.action(h,i,True),agent.action(altered,i,True),rtol=0,atol=0)
        state=torch.randn(4,5)
        self.assertFalse(torch.equal(agent.value(h,i,state),agent.value(h,i,state+3)))
        # The deployment action has no privileged argument whatsoever.
        self.assertTrue(set(map(id,agent.actor.parameters())).isdisjoint(map(id,agent.value.parameters())))

    def test_serialized_restart_and_on_policy_single_use(self):
        torch.manual_seed(12); agent=self.agent(); r=self.rollout(agent)
        first=agent.optimize(r)
        # Current diagnostics include nested trial records, not only scalars.
        def finite(value):
            if isinstance(value, dict): return all(finite(v) for v in value.values())
            if isinstance(value, (list, tuple)): return all(finite(v) for v in value)
            return value is None or math.isfinite(value)
        self.assertTrue(finite(first))
        self.assertGreater(first['optimizer_updates'],0)
        with self.assertRaises(ValueError): agent.optimize(r)
        agent.set_environment_steps(144000)
        buffer=io.BytesIO(); torch.save(agent.state_dict(),buffer); buffer.seek(0)
        other=self.agent(); other.load_state_dict(torch.load(buffer,weights_only=True))
        self.assertEqual(other.entropy_weight,agent.entropy_weight)
        r=self.rollout(agent)
        rng=torch.get_rng_state(); expected=agent.optimize(r)
        torch.set_rng_state(rng); actual=other.optimize(r)
        self.assertEqual(expected,actual)
        for x,y in zip(agent.actor.parameters(),other.actor.parameters()):
            torch.testing.assert_close(x,y,rtol=0,atol=0)
        for x,y in zip(agent.value.parameters(),other.value.parameters()):
            torch.testing.assert_close(x,y,rtol=0,atol=0)

    def _check_critic_finishes_when_actor_stops(self,device):
        agent=self.agent(device); r=self.rollout(agent)
        with torch.no_grad(): agent.actor.mean.bias.fill_(5)
        actor=deepcopy(agent.actor.state_dict())
        optimizer=deepcopy(agent.actor_optimizer.state_dict())
        value=deepcopy(agent.value.state_dict())
        result=agent.optimize(r)
        self.assertEqual(result['optimizer_updates'],0)
        self.assertEqual(result['value_optimizer_updates'],10)  # 9 valid samples, batches 8, 5 epochs.
        self.assertEqual(agent.value_updates,10)
        self.assertEqual(agent.actor_optimizer.state_dict(),optimizer)
        for k,x in actor.items(): torch.testing.assert_close(x,agent.actor.state_dict()[k],rtol=0,atol=0)
        self.assertTrue(any(not torch.equal(x,agent.value.state_dict()[k]) for k,x in value.items()))

    def test_critic_finishes_when_actor_stops(self):
        self._check_critic_finishes_when_actor_stops('cpu')

    def test_actor_stays_frozen_after_mid_epoch_kl_stop(self):
        agent=self.agent(); agent.settings=deepcopy(agent.settings)
        r=self.rollout(agent); captured=[]; step=agent.actor_optimizer.step
        def capture(*args,**kw):
            value=step(*args,**kw)
            captured.append((deepcopy(agent.actor.state_dict()),deepcopy(agent.actor_optimizer.state_dict())))
            return value
        agent.actor_optimizer.step=capture
        # Force the next minibatch's pre-step gate after ONE accepted current
        # bounded update; a tiny legacy threshold now rejects the first trial.
        original_kl=gaussian_kl
        def later_gate(*args):
            kl=original_kl(*args)
            return kl+1 if agent.updates else kl
        with patch('training.ppo.gaussian_kl',side_effect=later_gate):
            result=agent.optimize(r)
        self.assertTrue(result['kl_early_stop'])
        self.assertEqual(result['optimizer_updates'],1)
        self.assertEqual(result['value_optimizer_updates'],10)
        for k,x in captured[-1][0].items(): torch.testing.assert_close(x,agent.actor.state_dict()[k],rtol=0,atol=0)
        for k,state in captured[-1][1]['state'].items():
            for name,x in state.items(): torch.testing.assert_close(x,agent.actor_optimizer.state_dict()['state'][k][name],rtol=0,atol=0)

    def test_whole_cohort_diagnostics_and_padding(self):
        agent=self.agent(); r=self.rollout(agent)
        with torch.no_grad(): agent.actor.mean.bias.add_(.1)
        _,target,valid=advantages(r['rewards']*agent.settings['reward_scale'],r['values'],r['lengths'],1.,.95)
        b,t=valid.nonzero(as_tuple=True); targets=target[b,t]
        rng=torch.get_rng_state(); diag=agent.cohort_diagnostics(r,b,t,targets)
        torch.testing.assert_close(torch.get_rng_state(),rng,rtol=0,atol=0)
        with torch.no_grad():
            mean,logstd=agent.actor.distribution(r['frames'][b],t*r['stride'])
            ref=torch.distributions.kl_divergence(
                torch.distributions.Normal(r['mean'][b,t],r['log_std'][b,t].exp()),
                torch.distributions.Normal(mean,logstd.exp())).sum(-1).mean()
            pred=agent.value(r['frames'][b],t*r['stride'],r['privileged'][b,t])
        self.assertAlmostEqual(diag['policy_kl_mean'],float(ref),places=6)
        self.assertAlmostEqual(diag['value_explained_variance'],1-float((targets-pred).var(unbiased=False)/targets.var(unbiased=False)),places=6)
        self.assertAlmostEqual(diag['value_mse'],float((targets-pred).square().mean()),places=2)
        altered=target.clone(); altered[~valid]=1e9
        self.assertEqual(diag,agent.cohort_diagnostics(r,b,t,altered[b,t]))
        self.assertIsNone(explained_variance(torch.ones(3),torch.ones(3)))


class BoundedUpdateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.helper = PPOFixture()

    def agent(self, device='cpu', trials=8):
        agent = self.helper.agent(device)
        agent.settings = deepcopy(agent.settings)
        agent.settings['ppo']['post_step_kl'] = {'maximum_trials':trials}
        return agent

    def transaction(self, parameter, optimizer, criterion, trials=8):
        return bounded_actor_step([parameter],optimizer,criterion,.02,trials)

    def fixture(self):
        p = nn.Parameter(torch.zeros(1,dtype=torch.float64))
        opt = torch.optim.Adam([p],lr=.1)
        p.grad = torch.full_like(p,-1.)
        return p,opt

    def test_accept_backtrack_and_restore_nominal_rate(self):
        p,opt=self.fixture()
        outcome=self.transaction(p,opt,lambda:.5*(p.item()/.01)**2)
        self.assertTrue(outcome['accepted'])
        self.assertGreater(outcome['trials'][0]['kl'],.02)
        self.assertLessEqual(outcome['trials'][-1]['kl'],.02)
        self.assertEqual(int(opt.state[p]['step']),1)
        self.assertEqual(opt.param_groups[0]['lr'],.1)
        p,opt=self.fixture()
        self.assertEqual(self.transaction(p,opt,lambda:.5*p.item()**2)['factor'],1.)

    def test_total_rejection_empty_and_nonempty_adam(self):
        for initialized in (False,True):
            p,opt=self.fixture()
            if initialized: opt.step()
            old,parameter=deepcopy(opt.state_dict()),p.detach().clone()
            outcome=self.transaction(p,opt,lambda:1.,trials=2)
            self.assertFalse(outcome['accepted'])
            self.assertTrue(torch.equal(p,parameter))
            self.assertTrue(exact_state(old,opt.state_dict()))

    def test_nonfinite_rejection_and_exception_rollback(self):
        for kind in ('criterion','parameter','exception'):
            p,opt=self.fixture(); before=deepcopy(opt.state_dict())
            if kind=='parameter': p.grad.fill_(float('nan'))
            def criterion():
                if kind=='exception': raise RuntimeError('Deliberate trial exception')
                return float('nan') if kind=='criterion' else 0.
            if kind=='exception':
                with self.assertRaisesRegex(RuntimeError,'Deliberate'):
                    self.transaction(p,opt,criterion)
            else: self.assertFalse(self.transaction(p,opt,criterion)['accepted'])
            self.assertEqual(p.item(),0.)
            self.assertTrue(exact_state(before,opt.state_dict()))

    def check_integration(self,device):
        torch.manual_seed(33)
        if device=='cuda': torch.cuda.manual_seed_all(33)
        agent=self.agent(device); r=self.helper.rollout(agent)
        # Deliberately larger diagnostic rate forces post-step rejection/retry.
        for g in agent.actor_optimizer.param_groups: g['lr']=.03
        inputs=deepcopy(r); entropy=agent.entropy_weight
        result=agent.optimize(r)
        self.assertGreater(result['post_step_kl_trials'],0)
        self.assertLessEqual(result['post_step_kl_maximum_accepted'],.02)
        self.assertLessEqual(result['post_step_kl_final'],.02)
        self.assertEqual(result['value_optimizer_updates'],10)
        self.assertEqual(agent.value_updates,10)
        self.assertTrue(exact_state(r,inputs))
        self.assertEqual(agent.entropy_weight,entropy)
        self.assertEqual(agent.actor_optimizer.param_groups[0]['lr'],.03)
        for s in agent.actor_optimizer.state.values(): self.assertEqual(int(s['step']),agent.updates)
        return agent

    def test_cpu_full_optimize_cohort_bound_and_independent_critic(self):
        self.check_integration('cpu')

    def check_restart(self,device):
        agent=self.check_integration(device)
        stream=io.BytesIO(); torch.save(agent.state_dict(),stream); stream.seek(0)
        other=self.agent(device)
        other.load_state_dict(torch.load(stream,weights_only=True,map_location=device))
        r=self.helper.rollout(agent)
        cpu=torch.get_rng_state(); gpu=torch.cuda.get_rng_state_all() if device=='cuda' else None
        expected=agent.optimize(r)
        torch.set_rng_state(cpu)
        if gpu is not None: torch.cuda.set_rng_state_all(gpu)
        actual=other.optimize(r)
        self.assertEqual(expected,actual)
        self.assertTrue(exact_state(agent.state_dict(),other.state_dict()))

    def test_cpu_serialized_restart(self): self.check_restart('cpu')

    def test_rejected_production_step_restores_adam_and_finishes_value(self):
        agent=self.agent(trials=1); r=self.helper.rollout(agent)
        for g in agent.actor_optimizer.param_groups: g['lr']=10.
        before=deepcopy(agent.actor.state_dict()); opt=deepcopy(agent.actor_optimizer.state_dict())
        result=agent.optimize(r)
        self.assertEqual(result['optimizer_updates'],0)
        self.assertEqual(result['post_step_kl_rejected_updates'],1)
        self.assertTrue(exact_state(before,agent.actor.state_dict()))
        self.assertTrue(exact_state(opt,agent.actor_optimizer.state_dict()))
        self.assertEqual(result['value_optimizer_updates'],10)
        self.assertEqual(result['post_step_kl_final'],0.)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda_serialized_restart(self): self.check_restart('cuda')
