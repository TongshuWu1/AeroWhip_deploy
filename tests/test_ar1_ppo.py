"""Correlated PPO density, gradients, collection reset and optimizer contracts."""
from copy import deepcopy
import math
import unittest
import torch
from torch import nn
from training.ppo import PPO, ar1_distribution, squashed_log_prob, gaussian_kl
from training.configuration import load
from helpers import exact_state
from test_shape_release import PRESET


class AR1Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def agent(self,device='cpu'):
        s=load(PRESET)[0];s.update(device=device,batch_size=8,hidden_units=16)
        return PPO(s,[0,1,2],6,device,privileged_size=5,encoder_options=dict(goal_columns=[3,4,5]))

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
        with self.assertRaises(ValueError):a.initialize_actor(saved)
        self.assertEqual(a.interface['action_noise']['likelihood'],'history_conditional_squashed_gaussian_v1')

    def check_optimizer(self,device):
        torch.manual_seed(3);a=self.agent(device);r=self.rollout(a)
        out=a.optimize(r);self.assertGreater(out['optimizer_updates'],0)
        self.assertLessEqual(out['post_step_kl_final'],a.settings['ppo']['target_kl'])
        self.assertEqual(out['value_optimizer_updates'],10)
        other=self.agent(device);other.load_state_dict(deepcopy(a.state_dict()));r=self.rollout(a)
        rng=torch.get_rng_state();cuda=torch.cuda.get_rng_state() if device=='cuda' else None
        expected=a.optimize(r);torch.set_rng_state(rng)
        if cuda is not None:torch.cuda.set_rng_state(cuda)
        actual=other.optimize(r);self.assertEqual(expected,actual)
        self.assertTrue(exact_state(a.state_dict(),other.state_dict()))

    def test_optimizer_restart_and_post_step_kl(self):self.check_optimizer('cpu')

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable')
    def test_cuda_optimizer_restart_and_post_step_kl(self):self.check_optimizer('cuda')
