"""Captured forward and recomputed first-order gradients for recurrent fitting."""

import torch
from torch.autograd.function import once_differentiable


class CudaBlock:
    def __init__(self, function, examples, *, training, parameters=()):
        self.training = training
        self.parameters = tuple(parameters) if training else ()
        self.inputs = tuple(x.detach().clone(memory_format=torch.contiguous_format)
                            .requires_grad_(training and x.requires_grad)
                            for x in examples)
        self.differentiable = tuple(i for i, x in enumerate(self.inputs) if x.requires_grad)
        self.signature = tuple((x.shape, x.dtype, x.device) for x in examples)
        stream = torch.cuda.Stream(device=examples[0].device)
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                # Compilation may specialize on grad mode. Warm both paths
                # before entering capture, including inference-only kernels.
                with torch.no_grad():
                    outputs = tuple(function(*self.inputs))
                    del outputs
                if training:
                    with torch.enable_grad():
                        outputs = tuple(function(*self.inputs))
                        gradients = torch.autograd.grad(outputs,
                            tuple(self.inputs[i] for i in self.differentiable)+self.parameters,
                            tuple(torch.ones_like(x) for x in outputs))
                        del gradients
                        # Release the tape on its own stream before capture.
                        del outputs
        torch.cuda.current_stream().wait_stream(stream)
        self.forward_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.forward_graph, stream=stream), torch.no_grad():
            self.outputs = tuple(function(*self.inputs))
        if training:
            self.cotangents = tuple(torch.zeros_like(x) for x in self.outputs)
            self.backward_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.backward_graph, stream=stream), torch.enable_grad():
                outputs = tuple(function(*self.inputs))
                self.gradients = torch.autograd.grad(outputs,
                    tuple(self.inputs[i] for i in self.differentiable)+self.parameters, self.cotangents)

    def __call__(self, *inputs):
        if tuple((x.shape, x.dtype, x.device) for x in inputs) != self.signature:
            raise ValueError('Captured fitting input signature changed')
        if torch.is_grad_enabled() and not self.training:
            raise RuntimeError('Comparison-only graph cannot compute gradients')
        return _RecomputedBlock.apply(self, *inputs, *self.parameters)


class _RecomputedBlock(torch.autograd.Function):
    @staticmethod
    def forward(ctx, block, *inputs):
        ctx.block = block
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(*inputs)
        for target, value in zip(block.inputs, inputs[:len(block.inputs)]):
            target.copy_(value)
        block.forward_graph.replay()
        return tuple(value.clone() for value in block.outputs)

    @staticmethod
    @once_differentiable
    def backward(ctx, *cotangents):
        block = ctx.block
        for target, value in zip(block.inputs, ctx.saved_tensors):
            target.copy_(value)
        for target, value in zip(block.cotangents, cotangents):
            target.zero_() if value is None else target.copy_(value)
        block.backward_graph.replay()
        gradients = [None] * (len(block.inputs)+len(block.parameters))
        indices = block.differentiable + tuple(range(len(block.inputs),len(gradients)))
        for index, value in zip(indices, block.gradients):
            if ctx.needs_input_grad[index + 1]:
                gradients[index] = value.clone()
        return (None, *gradients)
