"""Bounded effective response corrections; not identified material forces."""

from copy import deepcopy
import hashlib
import math
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = 'coupled_residual_v2'


def _mlp(inputs, hidden):
    net = nn.Sequential(nn.Linear(inputs, hidden), nn.Tanh(),
                        nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 3))
    nn.init.zeros_(net[-1].weight)
    nn.init.zeros_(net[-1].bias)
    return net


class CoupledResidual(nn.Module):
    def __init__(self, specification):
        super().__init__()
        self.specification = deepcopy(specification)
        self.drone_equilibrium_centered = specification.get('drone_equilibrium_centered', False)
        if not isinstance(self.drone_equilibrium_centered, bool):
            raise ValueError('drone_equilibrium_centered must be boolean')
        hidden = specification['hidden_units']
        if not isinstance(hidden, int) or hidden < 1:
            raise ValueError('Positive integer residual width required')
        for key in ('drone_limit_m_s2', 'cable_limit_m_s2', 'position_scale_m',
                    'velocity_scale_m_s', 'acceleration_scale_m_s2',
                    'angular_velocity_scale_rad_s', 'motion_gate_speed_m_s'):
            value = float(specification[key])
            if not math.isfinite(value) or value <= 0:
                raise ValueError('Positive finite residual setting required: '+key)
            setattr(self, key, value)
        self.drone = _mlp(21, hidden)
        self.cable = _mlp(20, hidden)
        self.register_buffer('_hover_features',
            torch.tensor([0.]*15 + [0., 0., -1., 1., 0., 0.]), persistent=False)

    def drone_acceleration(self, model, state, command):
        p, v = model.tracking_pose(state)
        R = state.rotation
        heading = torch.stack((command[9].cos(), command[9].sin(), torch.zeros_like(command[9])))
        features = torch.cat((
            R.T @ (command[:3]-p) / self.position_scale_m,
            R.T @ (command[3:6]-v) / self.velocity_scale_m_s,
            R.T @ command[6:9] / self.acceleration_scale_m_s2,
            R.T @ state.velocity_com / self.velocity_scale_m_s,
            state.omega_body / self.angular_velocity_scale_rad_s,
            R.T @ (model.gravity/model.gravity.norm().clamp_min(1e-12)),
            R.T @ heading))
        response = self.drone(features)
        if self.drone_equilibrium_centered:
            # Level, yaw-aligned hover: zero tracking errors and motion,
            # body-frame gravity -Z and commanded heading +X. Subtract before
            # tanh so the original per-axis acceleration bound still holds.
            response = response - self.drone(self._hover_features)
        body = self.drone_limit_m_s2 * torch.tanh(response)
        return R @ body

    def cable_acceleration(self, model, state):
        q, v = model.cable_state(state)
        R = state.rotation
        edges = q[1:]-q[:-1]
        incoming = edges / edges.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        outgoing = torch.cat((incoming[1:], torch.zeros_like(incoming[:1])))
        root_velocity = v[:1].expand_as(v[1:])
        features = torch.cat((incoming @ R, outgoing @ R,
            (v[1:] @ R)/self.velocity_scale_m_s,
            ((q[1:]-q[:1]) @ R)/model.rest.sum(),
            ((v[1:]-v[:1]) @ R)/self.velocity_scale_m_s,
            model.arc_fraction[:,None], (root_velocity @ R)/self.velocity_scale_m_s,
            model.tip_mask[:,None]), dim=-1)
        motion = (v[1:].square().sum(-1,keepdim=True)
                  + root_velocity.square().sum(-1,keepdim=True))/self.motion_gate_speed_m_s**2
        gate = -torch.expm1(-motion)
        body = gate*self.cable_limit_m_s2*torch.tanh(self.cable(features))
        return body @ R.T

    def magnitude(self, model, state, command):
        drone = self.drone_acceleration(model, state, command)/self.drone_limit_m_s2
        cable = self.cable_acceleration(model, state)/self.cable_limit_m_s2
        return drone.square().mean() + cable.square().mean()


def save_residual(path, network):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    torch.save({'schema':SCHEMA, 'specification':network.specification,
                'state_dict':{k:v.detach().cpu() for k,v in network.state_dict().items()}}, temporary)
    temporary.replace(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_residual(reference, device):
    path = Path(reference['checkpoint'])
    if not path.is_absolute():
        path = ROOT/path
    if hashlib.sha256(path.read_bytes()).hexdigest() != reference['sha256']:
        raise ValueError('Residual checkpoint hash mismatch')
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if payload['schema'] not in ('coupled_residual_v1', SCHEMA):
        raise ValueError('Unsupported coupled residual checkpoint')
    network = CoupledResidual(payload['specification']).to(device=device, dtype=torch.float64)
    network.load_state_dict(payload['state_dict'])
    network.eval().requires_grad_(False)
    return network
