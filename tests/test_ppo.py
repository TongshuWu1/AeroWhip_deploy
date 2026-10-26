"""PPO: GAE, clipped surrogate, squashed densities, KL-bounded updates, AR(1) exploration
likelihood, entropy schedule and exact restart. Synthetic inputs; no plant."""
from copy import deepcopy
import io
import math
from types import SimpleNamespace
import unittest

import torch
from torch import nn

from helpers import EXPERIMENT, PPOFixture, exact_state, small_agent
from training.ppo import PPO, advantages, ar1_distribution, bounded_actor_step, clipped_loss, gaussian_kl, squashed_log_prob


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

class BoundedUpdateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.helper = PPOFixture()

    def agent(self, device='cpu', trials=8):
        return small_agent(device, batch_size=8, action_noise_correlation_s=None, post_step_kl_trials=trials)

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


class AR1Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def agent(self,device='cpu'):
        return small_agent(device,batch_size=8,hidden_units=16)

    def rollout(self,agent):
        dev=agent.device;f=torch.randn(4,7,6,device=dev);states=torch.randn(4,4,5,device=dev)
        r=dict(collection=agent.collections+1,stride=2,frames=f,privileged=states)
        previous=previous_index=None
        entries=[]
        for t in range(3):
            index=torch.full((4,),2*t,device=dev,dtype=torch.long)
            a,raw,lp,m,s,v=agent.collect_action(f,index,states[:,t],previous_raw=previous,previous_index=previous_index)
            entries.append((raw,lp,m,s,v));previous,previous_index=raw,index
        for j,key in enumerate(('raw','logp','mean','log_std','values')):r[key]=torch.stack([x[j] for x in entries],1)
        r.update(rewards=torch.tensor([[.1,.2,1.],[.3,-10,0],[0,.2,.5],[-10,0,0]],device=dev),lengths=torch.tensor([3,2,3,1],device=dev))
        return r

    def test_joint_gaussian_factorization_and_gradients(self):
        torch.manual_seed(32)
        rho=math.exp(-.2)
        mu=torch.tensor([.2,-.1,.7],dtype=torch.float64,requires_grad=True)
        ls=torch.tensor([-.4,.2,-.8],dtype=torch.float64,requires_grad=True)
        raw=torch.tensor([.8,-.3,1.],dtype=torch.float64)
        t=torch.arange(3);prev=(t-1).clamp_min(0)
        m,s=ar1_distribution(mu[:,None],ls[:,None],mu[prev,None],ls[prev,None],raw[prev,None],t==0,rho)
        lp=squashed_log_prob(raw[:,None],m,s).sum()
        cov=ls.exp()[:,None]*ls.exp()[None,:]*rho**(t[:,None]-t[None,:]).abs().double()
        expected=torch.distributions.MultivariateNormal(mu,covariance_matrix=cov).log_prob(raw)-torch.log1p(-raw.tanh().square()).sum()
        torch.testing.assert_close(lp,expected,rtol=1e-12,atol=1e-12)
        g=torch.autograd.grad(lp,(mu,ls),retain_graph=True);other=torch.autograd.grad(expected,(mu,ls))
        for a,b in zip(g,other):torch.testing.assert_close(a,b,rtol=1e-11,atol=1e-11)
        # Conditional t=1 loss must also differentiate its previous-policy mean.
        g=torch.autograd.grad(m[1].sum(),mu)[0];self.assertNotEqual(float(g[0]),0)

    def test_population_stationarity_and_correlation(self):
        torch.manual_seed(7);rho=math.exp(-.2);sigma=math.exp(-.3)/2
        eps=torch.randn(100000,dtype=torch.float64);start=eps.clone()
        for _ in range(5):eps=rho*eps+math.sqrt(1-rho*rho)*torch.randn_like(eps)
        self.assertAlmostEqual(float((sigma*eps).std()),sigma,delta=.003)
        self.assertAlmostEqual(float(torch.corrcoef(torch.stack((start,eps)))[0,1]),rho**5,delta=.01)

    def test_behavior_probabilities_recomputed_and_change_with_previous_policy(self):
        torch.manual_seed(1);a=self.agent();r=self.rollout(a)
        b,t=(torch.arange(3)[None,:]<r['lengths'][:,None]).nonzero(as_tuple=True)
        m,s=a.rollout_distribution(r,b,t)
        torch.testing.assert_close(m,r['mean'][b,t]);torch.testing.assert_close(s,r['log_std'][b,t])
        torch.testing.assert_close(squashed_log_prob(r['raw'][b,t],m,s),r['logp'][b,t])
        self.assertAlmostEqual(a.cohort_policy_kl(r,b,t),0,places=12)
        self.assertIsNone(a.prepare_kl_inputs(r,b,t))
        with torch.no_grad():a.actor.mean.bias.add_(.2)
        changed,_=a.rollout_distribution(r,b,t)
        shift=torch.where(t[:,None]==0,.2,.2*(1-a.noise_rho)).expand_as(m)
        torch.testing.assert_close(changed-m,shift)
        expected=gaussian_kl(r['mean'][b,t].double(),r['log_std'][b,t].double(),changed.double(),s.double()).mean()
        self.assertAlmostEqual(a.cohort_policy_kl(r,b,t),float(expected.detach()),places=10)

    def test_episode_reset_determinism_and_interface_rejection(self):
        a=self.agent();r=self.rollout(a);i=torch.zeros(4,dtype=torch.long)
        rng=torch.get_rng_state();one=a.collect_action(r['frames'],i,r['privileged'][:,0])
        a.action(r['frames'],i,True);torch.set_rng_state(rng)
        two=a.collect_action(r['frames'],i,r['privileged'][:,0])
        for x,y in zip(one,two):torch.testing.assert_close(x,y,rtol=0,atol=0)
        with self.assertRaises(ValueError):a.collect_action(r['frames'],i+2,r['privileged'][:,1])
        saved=deepcopy(a.state_dict());saved['interface'].pop('action_noise')
        with self.assertRaises(ValueError):a.load_state_dict(saved)
        self.assertEqual(a.interface['action_noise']['likelihood'],'history_conditional_squashed_gaussian_v1')


class ExponentialEntropyTests(unittest.TestCase):
    def weight(self, episodes, *, budget=12800):
        a=SimpleNamespace(config=EXPERIMENT.ppo,completed_episodes=episodes,entropy_episode_budget=budget)
        return PPO.entropy_weight.fget(a)

    def test_approved_curve_and_fixed_episode_timescale(self):
        for n in (0,1,20,40,60,81,120):
            expected=.001+.019*math.exp(-8*n/200)
            self.assertAlmostEqual(self.weight(n*64),expected,places=15)
            self.assertEqual(self.weight(n*64),self.weight(n*128,budget=25600))

    def test_fast_early_decrease_then_flatten_with_positive_floor(self):
        values=[self.weight(n*64) for n in range(121)]
        drops=[a-b for a,b in zip(values,values[1:])]
        self.assertTrue(all(d>0 for d in drops))
        self.assertTrue(all(a>b for a,b in zip(drops,drops[1:])))
        self.assertGreater(values[-1],.001)
        self.assertEqual(self.weight(10**9),.001)
