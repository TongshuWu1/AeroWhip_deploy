"""Separated aerodynamic coefficient learners; physical forces live in C++."""

from copy import deepcopy
import hashlib
import math

import torch
from torch import nn
from aerowhip.paths import project_path
from aerowhip.fitting.artifacts import atomic_save

SCHEMA = 'structured_aerodynamic_residual_v1'


def _mlp(inputs, hidden, outputs):
    net = nn.Sequential(nn.Linear(inputs, hidden), nn.Tanh(),
                        nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, outputs))
    nn.init.zeros_(net[-1].weight)
    nn.init.zeros_(net[-1].bias)
    return net


class AerodynamicResidual(nn.Module):
    def __init__(self, specification):
        super().__init__()
        self.specification = deepcopy(specification)
        hidden = specification['hidden_units']
        expected = {'hidden_units','cable_log_bound','drone_max_rate_s_inv','velocity_scale_m_s',
                    'drone_initial_logit','cable_enabled','drone_enabled'}
        if set(specification) != expected:
            raise ValueError('Expected structured aerodynamic specification; no legacy weight migration')
        if not isinstance(hidden, int) or isinstance(hidden, bool) or hidden < 1:
            raise ValueError('Positive integer residual width required')
        for key in ('cable_log_bound', 'drone_max_rate_s_inv', 'velocity_scale_m_s'):
            value = float(specification[key])
            if not math.isfinite(value) or value <= 0:
                raise ValueError('Positive finite residual setting required: '+key)
            setattr(self, key, value)
        if self.cable_log_bound > 10 or not math.isfinite(specification['drone_initial_logit']):
            raise ValueError('Invalid aerodynamic coefficient prior')
        for key in ('cable_enabled','drone_enabled'):
            if not isinstance(specification[key], bool):
                raise ValueError('Explicit boolean branch status required')
        self.cable = _mlp(2, hidden, 2)
        self.drone = _mlp(3, hidden, 3)
        nn.init.constant_(self.drone[-1].bias, specification['drone_initial_logit'])
        self.completed_stages = []
        self.fitting_stage = None

    def select_stage(self, stage):
        if stage not in ('cable','drone'):
            raise ValueError('Fit cable first, then drone; no joint residual stage')
        if stage == 'drone' and 'cable' not in self.completed_stages:
            raise ValueError('Freeze a gradient-checked cable fit before fitting drone drag')
        self.specification.update(cable_enabled=True, drone_enabled=stage == 'drone')
        self.requires_grad_(False)
        getattr(self, stage).requires_grad_(True)
        self.fitting_stage = stage

    def tensor_parameters(self):
        return ([v for branch in (self.cable,self.drone) for i in (0,2,4)
                 for v in (branch[i].weight,branch[i].bias)],
                [self.cable_log_bound,self.drone_max_rate_s_inv,self.velocity_scale_m_s,
                 float(self.specification['cable_enabled']),float(self.specification['drone_enabled'])])

    def magnitude(self, model, state, command):
        cable, drone = model.aerodynamic_accelerations(state, self)
        if self.fitting_stage == 'drone':
            return (drone/(self.drone_max_rate_s_inv*self.velocity_scale_m_s)).square().mean()
        _, velocity = model.cable_state(state)
        delta = cable + model.drag*velocity
        scale = model.drag*math.exp(self.cable_log_bound)*self.velocity_scale_m_s
        return (delta/scale.clamp_min(1e-12)).square().mean()


def save_residual(path, network):
    path = project_path(path)
    payload={'schema':SCHEMA, 'specification':network.specification,
             'completed_stages':network.completed_stages,
             'state_dict':{k:v.detach().cpu() for k,v in network.state_dict().items()}}
    atomic_save(path,lambda temporary:torch.save(payload,temporary))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_residual(reference, device):
    path = project_path(reference['checkpoint'])
    if hashlib.sha256(path.read_bytes()).hexdigest() != reference['sha256']:
        raise ValueError('Residual checkpoint hash mismatch')
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if payload['schema'] != SCHEMA:
        raise ValueError('Residual checkpoint is not a structured aerodynamic model; no legacy migration')
    network = AerodynamicResidual(payload['specification']).to(device=device, dtype=torch.float64)
    network.load_state_dict(payload['state_dict'])
    network.completed_stages = payload['completed_stages']
    if network.completed_stages not in ([], ['cable'], ['cable','drone']):
        raise ValueError('Invalid residual fitting-stage history')
    if any(not bool(torch.isfinite(p).all()) for p in network.parameters()):
        raise ValueError('Nonfinite aerodynamic weights')
    network.eval().requires_grad_(False)
    return network
