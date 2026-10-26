"""GPU-controlled Picard loop around the unchanged compiled slot, for batched environments."""
import torch
from types import MethodType
from .conditional_driver import Driver
from .iteration_graph import _static_like,_reduced
from simulation.tensor_model import ops

class ConditionalSlots:
    def __init__(self,kernel,driver):
        self.kernel,self.driver=kernel,driver
        self.cache={}
        self.counters=None
    def __call__(self,state,parameters,wrench,h,iterations,weights,specification,work):
        if torch.is_grad_enabled():raise RuntimeError('Conditional task solver cannot replace differentiable physics')
        if not all(t.device==wrench.device for t in state+list(work)):
            raise ValueError("Conditional solver buffers must share one CUDA device")
        values=state+[wrench]+list(work)
        key=(h,iterations,tuple((tuple(t.shape),t.stride(),t.dtype) for t in values))
        if key not in self.cache:
            first=[_static_like(t) for t in values]
            def invoke(buffers):
                return self.kernel(buffers[:9],parameters,buffers[9],h,iterations,weights,specification,buffers[10:])
            stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                initial=invoke(first);invoke(first)
                # Separate zero-stride startup vs contiguous later specialization.
                loop=first[:10]+[_static_like(t) for t in initial]
                invoke(loop);invoke(loop)
            torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
            assert loop[11].dtype==torch.bool and loop[11].stride()==(1,)
            def copy_work(result):
                for buffer,value in zip(loop[10:],result):_reduced(buffer).copy_(_reduced(value))
            first_graph=torch.cuda.CUDAGraph(keep_graph=True)
            loop_graph=torch.cuda.CUDAGraph(keep_graph=True)
            with torch.compiler.set_stance('fail_on_recompile'):
                with torch.cuda.graph(first_graph,stream=stream):copy_work(invoke(first))
                with torch.cuda.graph(loop_graph,stream=stream):copy_work(invoke(loop))
            count=torch.empty(1,device=wrench.device,dtype=torch.int32)
            graph,exec=self.driver.loop(first_graph,loop_graph,loop[11],count,loop[11].numel())
            self.cache[key]=dict(first=first,work=loop[10:],first_graph=first_graph,loop_graph=loop_graph,
                                 count=count,graph=graph,exec=exec)
        entry=self.cache[key]
        for buffer,value in zip(entry['first'],values):_reduced(buffer).copy_(_reduced(value))
        self.driver.launch(entry['exec'])
        # Capture counters without a readback; optional diagnostic copies only.
        if self.counters is not None:self.counters.append(entry['count'].clone())
        return entry['work']
    def close(self):
        if not self.cache:return
        torch.cuda.synchronize()
        for entry in self.cache.values():
            self.driver.call('cuGraphExecDestroy',entry['exec'])
            self.driver.call('cuGraphDestroy',entry['graph'])
        self.cache.clear()

    def __del__(self):
        try:self.close()
        except Exception:pass  # Shutdown only; explicit close reports errors.


def install(model,driver=None):
    """Replace only this model instance's iteration scheduling."""
    m=model
    if not m.rest.is_cuda or not m.batched:
        raise ValueError("Conditional task replay needs a CUDA batched model")
    m.compile_batched_physics()
    slots=ConditionalSlots(m._batch_implicit_iteration,Driver() if driver is None else driver)
    def substep(self,state,initial,target,actuator,parameters,h,start,iterations,weights,specification):
        wrench=torch.vmap(lambda a,b:ops.motor_wrench(
            ops.motor_response_elapsed(a,b,actuator,start+h),actuator))(initial,target)
        work=torch.vmap(lambda *s:ops.implicit_begin(list(s)))(*state)
        work=slots(state,parameters,wrench,h,iterations,weights,specification,work)
        return torch.vmap(lambda *v:ops.implicit_finish(list(v[:9]),list(v[9:]),h))(*state,*work)
    m._batch_motor_substep=MethodType(substep,m)
    return slots
