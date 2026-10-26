"""CPU execution of independent cable segments with one global optimizer update."""
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import numpy as np
import torch

from aerowhip.fitting.coupled_fit import window_weights
from aerowhip.fitting.prescribed_residual_objective import PrescribedResidualObjective
from aerowhip.fitting.residual_network import AerodynamicResidual

_worker = None


def _initialize(config, windows, specification, penalty, weights):
    global _worker
    torch.set_num_threads(1)
    network = AerodynamicResidual(specification).double()
    network.select_stage('cable')
    _worker = (config, windows, network, penalty, weights)


def _segment(index, state, backward):
    config, windows, network, penalty, weights = _worker
    network.load_state_dict({key:torch.from_numpy(value) for key,value in state.items()})
    network.zero_grad(set_to_none=True)
    objective = PrescribedResidualObjective(config, [windows[index]], network, penalty)
    # Preserve the full dataset's weights; do not normalize a one-segment task.
    objective.weights = [weights[index]]
    value = objective.loss(backward)
    gradient = None
    if backward:
        gradient = []
        for parameter in network.cable.parameters():
            if parameter.grad is None or not bool(torch.isfinite(parameter.grad).all()):
                raise FloatingPointError('Missing/nonfinite parallel cable gradient')
            gradient.append(parameter.grad.detach().numpy().copy())
    return index, value, gradient


class ParallelPrescribedResidualObjective(PrescribedResidualObjective):
    """Same per-segment loss; ordered global gradient accumulation on the parent.

    Workers never optimize, clip gradients, change sampling or share evolving
    states. All segments complete before one parent clipping/optimizer step.
    """
    def __init__(self, config, windows, network, magnitude_penalty, workers, progress=None):
        if not isinstance(workers,int) or isinstance(workers,bool) or workers < 1:
            raise ValueError('Positive integer cable worker count required')
        if next(network.parameters()).device.type != 'cpu' or network.fitting_stage != 'cable':
            raise ValueError('Parallel prescribed fitting requires a CPU cable-stage network')
        self.network, self.windows, self.progress = network, windows, progress
        self.weights = window_weights(windows)
        self.batch_size = 1
        self.workers = min(workers,len(windows))
        self.pool = ProcessPoolExecutor(max_workers=self.workers,
            mp_context=multiprocessing.get_context('spawn'),initializer=_initialize,
            initargs=(config,windows,network.specification,magnitude_penalty,self.weights))

    def __enter__(self): return self

    def __exit__(self, *args):
        self.pool.shutdown(wait=True,cancel_futures=True)

    def loss(self, backward=False):
        state = {key:value.detach().numpy().copy() for key,value in self.network.state_dict().items()}
        futures = [self.pool.submit(_segment,index,state,backward) for index in range(len(self.windows))]
        results = {}
        try:
            for future in as_completed(futures):
                index,value,gradient = future.result()
                if not np.isfinite(value): raise FloatingPointError('Nonfinite parallel cable loss')
                results[index] = (value,gradient)
                if self.progress:
                    self.progress(dict(phase='loss_pass',stage='cable',backward=backward,
                        completed_segments=len(results),total_segments=len(self.windows),
                        execution_batch_size=1,execution_workers=self.workers))
        except BaseException:
            for future in futures: future.cancel()
            raise
        # Completion order must not change floating-point reduction order.
        total = 0.
        for index in range(len(self.windows)):
            value,gradient = results[index]; total += value
            if backward:
                for parameter,array in zip(self.network.cable.parameters(),gradient):
                    g = torch.from_numpy(array)
                    if parameter.grad is None: parameter.grad = g.clone()
                    else: parameter.grad.add_(g)
        return total
