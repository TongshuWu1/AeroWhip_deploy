"""CUDA-graph replay of unchanged compiled Picard slots for batched environments.

The simulator's existing host convergence test and20-slot cap remain outside
each graph. No equations, solver acceptance or differentiable fitting changes.
"""
import os
import struct
import sys

import torch


def _reduced(tensor):
    # Broadcast startup masks must retain zero strides: layout selects the
    # compiled specialization, and writes to overlapping expansions are illegal.
    strides=tensor.stride()
    if 0 not in strides:
        return tensor
    return tensor[tuple(slice(0,1) if stride==0 else slice(None)
                        for stride in strides)]


def _static_like(tensor):
    reduced=_reduced(tensor)
    base=torch.empty_strided(reduced.shape,reduced.stride(),
        dtype=tensor.dtype,device=tensor.device)
    base.copy_(reduced)
    return base.expand(tensor.shape)


class IterationGraphs:
    """One model/stream owner, no autograd; buffers live with their captured graph."""
    def __init__(self,kernel):
        self.kernel=kernel
        self.cache={}

    def __call__(self,state,parameters,wrench,h,iterations,weights,specification,work):
        if torch.is_grad_enabled():
            raise RuntimeError('Task iteration replay cannot replace differentiable physics')
        values=state+[wrench]+list(work)
        key=(h,iterations,tuple((tuple(t.shape),t.stride(),t.dtype) for t in values))
        if key not in self.cache:
            buffers=[_static_like(t) for t in values]
            stream=torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            def invoke():
                return self.kernel(buffers[:9],parameters,buffers[9],h,iterations,
                                   weights,specification,buffers[10:])
            # Both compile/layout preparation calls occur BEFORE capture.
            with torch.cuda.stream(stream):
                for _ in range(2): invoke()
            torch.cuda.current_stream().wait_stream(stream)
            torch.cuda.synchronize()
            graph=torch.cuda.CUDAGraph()
            with torch.compiler.set_stance('fail_on_recompile'),torch.cuda.graph(graph,stream=stream):
                result=invoke()
            # Destination layouts never change during this graph's lifetime.
            # Keep their reduced views; avoid rebuilding full-rank slice views
            # for every tensor on every Picard slot. Copies/order stay identical.
            self.cache[key]=([_reduced(t) for t in buffers],graph,result)
        destinations,graph,result=self.cache[key]
        for buffer,value in zip(destinations,values): buffer.copy_(_reduced(value))
        graph.replay()
        return result


def conditional_solver_enabled(conditional=None):
    """Supported 64-bit CUDA default; an explicit choice wins over the environment variable."""
    if conditional is not None:
        return conditional
    cuda_version = tuple(int(v) for v in (torch.version.cuda or '0.0').split('.')[:2])
    supported = os.name == 'nt' or (sys.platform == 'linux' and cuda_version >= (12,8))
    default = '1' if supported and struct.calcsize('P') == 8 else '0'
    flag = os.environ.get('AEROWHIP_CONDITIONAL_SOLVER', default)
    if flag not in ('0', '1'):
        raise ValueError('AEROWHIP_CONDITIONAL_SOLVER must be 0 or 1')
    return flag == '1'


def accelerate(model, conditional=None):
    """Compile one Picard slot and replay it with CUDA graphs; returns the backend name.

    Solver equations, convergence masks and the 20-slot cap are unchanged.
    """
    model.compile_batched_physics()
    if conditional_solver_enabled(conditional):
        from .conditional_graph import install
        return 'cuda_conditional_substep_guard_v1', install(model)
    model._batch_implicit_iteration = IterationGraphs(model._batch_implicit_iteration)
    return 'cuda_iteration_substep_guard_v1', None

