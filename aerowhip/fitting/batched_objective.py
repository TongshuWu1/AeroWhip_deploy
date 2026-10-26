"""Independent flight windows evaluated together through the shared integrator."""

from dataclasses import fields
import numpy as np
import torch
from torch.utils.checkpoint import checkpoint

from aerowhip.physics.tensor_model import TensorModel, ControlledState as PlantState
from aerowhip.control.tensor_controller import GeometricController
from aerowhip.fitting.cuda_block import CudaBlock
from aerowhip.fitting.coupled_fit import NAMES


class BatchedObjective:
    def __init__(self, config, windows, weights, device, *, graph=False, compile_physics=False,
                 training=True, block_steps=1, residual=None, magnitude_penalty=0., loss_scope='vehicle'):
        if not windows or block_steps < 1:
            raise ValueError('Provide flight windows and a positive block size')
        if len({len(w.applied_commands) for w in windows}) != 1:
            raise ValueError("Batched fitting requires equal window lengths; no padding or truncation")
        self.model = TensorModel(config, device=device, batched=True, residual=residual)
        self.magnitude_penalty = float(magnitude_penalty)
        if loss_scope not in ('coupled','vehicle'):
            raise ValueError('Coupled fitting or vehicle-only residual loss required')
        self.loss_scope = loss_scope
        self.controller = GeometricController(self.model)
        m, c = self.model, self.controller
        self.base_parameters = (c.position_gain_s_inv2, c.velocity_gain_s_inv,
                                c.attitude_frequency_rad_s)
        self.set_windows(windows, weights)
        self.marker_indices = torch.tensor(m.marker_indices[1:], device=device, dtype=torch.long)
        self.graph = graph
        self.training = training
        self.block_steps = min(block_steps, len(self.commands))
        if compile_physics:
            m.compile_batched_physics()
        self.blocks = {}
        if graph:
            if not str(device).startswith('cuda'):
                raise ValueError('Captured fitting requires CUDA')
            torch.backends.cuda.preferred_linalg_library('cusolver')
            sizes = {min(self.block_steps, len(self.commands)-k)
                     for k in range(0, len(self.commands), self.block_steps)}
            for size in sorted(sizes):
                # Each fitting block starts at a controller tick: incoming held
                # wrench is overwritten before integration, so its gradient is
                # structurally absent. Motor-state gradients remain necessary.
                samples = tuple(v.detach().clone().requires_grad_(training and i < len(self.initial)-1)
                                for i,v in enumerate(self.initial))
                samples += (self.commands[:size], m.tensor([0.]*len(NAMES)).requires_grad_(training),
                            self.positions[:,:size], self.markers[:,:size], self.rotations[:,:size])
                parameters = tuple(p for p in m.residual.parameters() if p.requires_grad) if m.residual is not None else ()
                self.blocks[size] = CudaBlock(self._block, samples, training=training, parameters=parameters)

    def set_windows(self, windows, weights):
        """Replace independent inputs at the same captured shape.

        Finish backward before replacing inputs. Weights are supplied by the
        caller and never renormalized: execution chunks can retain full-dataset
        weighting. No padding, truncation or recurrent-state transfer occurs.
        """
        if not windows or len(weights) != len(windows):
            raise ValueError('Provide one weight per independent window')
        lengths = {len(w.applied_commands) for w in windows}
        if len(lengths) != 1:
            raise ValueError('Equal window lengths required, without padding or truncation')
        if hasattr(self, 'commands') and (len(windows) != len(self.weights)
                or next(iter(lengths)) != len(self.commands)):
            raise ValueError('Cannot replace captured inputs with another batch/time shape')
        m = self.model
        states = [m.controller_state(type(w.initial)(*(m.tensor(getattr(w.initial,f.name))
                  for f in fields(w.initial)))) for w in windows]
        self.initial = tuple(torch.stack([getattr(s, f.name) for s in states])
                             for f in fields(PlantState))
        self.commands = m.tensor(np.stack([w.applied_commands for w in windows])).transpose(0,1)
        self.positions = m.tensor(np.stack([w.tracking_positions for w in windows]))
        self.markers = m.tensor(np.stack([w.cable_markers for w in windows]))
        self.rotations = m.tensor(np.stack([w.tracking_rotations for w in windows]))
        self.weights = m.tensor(weights)

    def _advance(self, *args):
        return self._transition(*args)

    def _transition(self, *args):
        m, c = self.model, self.controller
        state, command, log_scales = args[:-2], args[-2], args[-1]
        parameters = tuple(v*s for v,s in zip(self.base_parameters, log_scales.exp()))
        (c.position_gain_s_inv2, c.velocity_gain_s_inv, c.attitude_frequency_rad_s) = parameters

        try:
            out = m.step_unchecked(PlantState(*state),command,c,batch=True)
            return tuple(getattr(out,f.name) for f in fields(out))
        finally:
            (c.position_gain_s_inv2, c.velocity_gain_s_inv, c.attitude_frequency_rad_s) = self.base_parameters

    def _block(self, *args):
        m = self.model
        count = len(fields(PlantState))
        state = args[:count]
        commands, log_scales, positions, markers, rotations = args[count:]
        observe = torch.vmap(lambda *args: (m.tracking_pose(PlantState(*args))[0],
                                           m.cable_state(PlantState(*args))[0]))
        loss = torch.zeros_like(self.weights)
        ps, qs, rs = [], [], []
        finite = torch.ones_like(loss, dtype=torch.bool)
        for k, command in enumerate(commands):
            state = self._advance(*state, command, log_scales)
            for value in state:
                finite = finite & torch.isfinite(value).flatten(1).all(-1)
            p, q = observe(*state)
            q = q[:,self.marker_indices]
            loss = loss + ((p-positions[:,k])/.02).square().mean(-1)
            if self.loss_scope == 'coupled':
                loss = loss + ((q-markers[:,k])/.03).square().mean((-2,-1))
            loss = loss + ((state[1]-rotations[:,k])/.05).square().sum((-2,-1))/2
            if m.residual is not None and self.magnitude_penalty:
                penalty = torch.vmap(lambda *values: m.residual.magnitude(
                    m,PlantState(*values[:-1]),values[-1]))(*state,command)
                loss = loss + self.magnitude_penalty*penalty
            ps.append(p); qs.append(q); rs.append(state[1])
        loss = torch.where(finite, loss, torch.full_like(loss, float('nan')))
        return (*state, loss, torch.stack(ps,1), torch.stack(qs,1), torch.stack(rs,1))

    def rollout(self, log_scales, *, predictions=False):
        if log_scales.shape != (len(NAMES),):
            raise ValueError(f'Expected {len(NAMES)} log-scales in order {NAMES}')
        if torch.is_grad_enabled() and not self.training:
            raise RuntimeError('Comparison-only objective cannot compute gradients')
        state = tuple(v.detach().clone() for v in self.initial)
        loss = torch.zeros_like(self.weights)
        outputs = []
        for k in range(0, len(self.commands), self.block_steps):
            end = min(k+self.block_steps, len(self.commands))
            args = (*state, self.commands[k:end], log_scales, self.positions[:,k:end],
                    self.markers[:,k:end], self.rotations[:,k:end])
            if self.graph:
                result = self.blocks[end-k](*args)
            elif torch.is_grad_enabled():
                result = checkpoint(self._block, *args, use_reentrant=False)
            else:
                result = self._block(*args)
            count = len(fields(PlantState))
            state = result[:count]
            loss = loss + result[count]
            if predictions:
                outputs.append(result[count+1:])
        if predictions:
            return loss/len(self.commands), tuple(torch.cat(values,1) for values in zip(*outputs))
        return loss/len(self.commands)

    @torch.no_grad()
    def predict(self, values):
        losses, outputs = self.rollout(self.model.tensor(values), predictions=True)
        if not bool(torch.isfinite(losses).all()):
            raise FloatingPointError('Nonfinite batched replay state or loss')
        return tuple(value.cpu().numpy() for value in outputs)

    def evaluate(self, values, backward=False):
        x = self.model.tensor(values).detach().clone().requires_grad_(backward)
        with torch.set_grad_enabled(backward):
            losses = self.rollout(x)
            if not bool(torch.isfinite(losses).all()):
                raise FloatingPointError("Nonfinite batched fitting loss")
            total = (losses*self.weights).sum()
            if backward:
                total.backward()
                if x.grad is None or not bool(torch.isfinite(x.grad).all()):
                    raise FloatingPointError("Nonfinite batched fitting gradient")
        return float(total.detach()), x.grad.detach().cpu() if backward else None
