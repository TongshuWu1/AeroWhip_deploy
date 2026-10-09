"""Windows CUDA conditional graphs; PyTorch-bundled NVRTC, no build.

ABI layouts follow CUDA12.8+ cuda.h. No physical arithmetic runs in this helper.
This is a single-context/stream owner; failures are explicit, not eager fallback.
"""
import ctypes as C
import os
from pathlib import Path
import torch

SOURCE=r'''
typedef unsigned long long cudaGraphConditionalHandle;
extern "C" __device__ void cudaGraphSetConditional(cudaGraphConditionalHandle,unsigned int);
extern "C" __global__ void smoke(cudaGraphConditionalHandle handle,int *count,int cap) {
    ++*count;
    cudaGraphSetConditional(handle,*count<cap);
}
extern "C" __global__ void condition(cudaGraphConditionalHandle handle,int *count,
                                    const bool *done,int lanes,int increment) {
    if (increment) ++*count;
    else *count=1;
    bool all=true;
    for(int i=0;i<lanes;++i) all=all && done[i];
    cudaGraphSetConditional(handle,!all && *count<20);
}
'''

class Conditional(C.Structure):
    _fields_=[('handle',C.c_ulonglong),('type',C.c_uint),('size',C.c_uint),
              ('graphs',C.POINTER(C.c_void_p)),('context',C.c_void_p)]
class ParamsUnion(C.Union):
    _fields_=[('reserved',C.c_longlong*29),('conditional',Conditional)]
class NodeParams(C.Structure):
    _fields_=[('type',C.c_uint),('reserved0',C.c_int*3),('params',ParamsUnion),('reserved2',C.c_longlong)]
class KernelParams(C.Structure):
    _fields_=[('func',C.c_void_p),('gx',C.c_uint),('gy',C.c_uint),('gz',C.c_uint),
              ('bx',C.c_uint),('by',C.c_uint),('bz',C.c_uint),('shared',C.c_uint),
              ('args',C.POINTER(C.c_void_p)),('extra',C.POINTER(C.c_void_p))]

class Driver:
    def __init__(self,out=None):
        if os.name!="nt" or C.sizeof(C.c_void_p)!=8:
            raise RuntimeError("Conditional solver currently requires64-bit Windows")
        assert C.sizeof(NodeParams)==256 and C.sizeof(Conditional)==32
        self.directory=os.add_dll_directory(str(Path(torch.__file__).parent/'lib'))
        self.driver=C.WinDLL('nvcuda.dll')
        self.nvrtc=C.CDLL(str(Path(torch.__file__).parent/'lib/nvrtc64_120_0.dll'))
        self._bind()
        version=C.c_int();self.call("cuDriverGetVersion",C.byref(version))
        self.driver_version=version.value
        if version.value<12080:raise RuntimeError("Conditional solver requires CUDA driver API12.8+")
        a,b=C.c_int(),C.c_int();self.rtc("nvrtcVersion",C.byref(a),C.byref(b))
        self.nvrtc_version=(a.value,b.value)
        torch.cuda.init()
        self._context_owner=torch.empty(1,device='cuda')
        self.device=self._context_owner.device
        self.context=C.c_void_p();self.call('cuCtxGetCurrent',C.byref(self.context))
        assert self.context.value is not None,'PyTorch CUDA context is not current'
        program=C.c_void_p()
        self.rtc('nvrtcCreateProgram',C.byref(program),SOURCE.encode(),b'conditional.cu',0,None,None)
        major,minor=torch.cuda.get_device_capability()
        options=[f'--gpu-architecture=compute_{major}{minor}'.encode()]
        rc=self.nvrtc.nvrtcCompileProgram(program,len(options),(C.c_char_p*len(options))(*options))
        size=C.c_size_t();self.rtc('nvrtcGetProgramLogSize',program,C.byref(size))
        log=C.create_string_buffer(size.value);self.rtc('nvrtcGetProgramLog',program,log)
        if out is not None:(out/'nvrtc.log').write_text(log.value.decode(),encoding='utf-8')
        if rc:raise RuntimeError('NVRTC compilation failed: '+log.value.decode())
        self.rtc('nvrtcGetPTXSize',program,C.byref(size))
        ptx=C.create_string_buffer(size.value);self.rtc('nvrtcGetPTX',program,ptx)
        if out is not None:(out/'conditional.ptx').write_bytes(ptx.value)
        self.module=C.c_void_p();self.call('cuModuleLoadData',C.byref(self.module),ptx)
        self.functions={}
        for name in ('smoke','condition'):
            f=C.c_void_p();self.call('cuModuleGetFunction',C.byref(f),self.module,name.encode());self.functions[name]=f
        self.rtc('nvrtcDestroyProgram',C.byref(program))

    def _bind(self):
        P=C.c_void_p;I=C.c_uint;S=C.c_size_t;H=C.c_ulonglong
        bindings={
          'cuDriverGetVersion':[C.POINTER(C.c_int)],'cuModuleUnload':[P],
          'cuCtxGetCurrent':[C.POINTER(P)],'cuModuleLoadData':[C.POINTER(P),P],
          'cuModuleGetFunction':[C.POINTER(P),P,C.c_char_p],
          'cuGraphCreate':[C.POINTER(P),I],
          'cuGraphConditionalHandleCreate':[C.POINTER(H),P,P,I,I],
          'cuGraphAddNode':[C.POINTER(P),P,C.POINTER(P),S,C.POINTER(NodeParams)],
          'cuGraphAddKernelNode':[C.POINTER(P),P,C.POINTER(P),S,C.POINTER(KernelParams)],
          'cuGraphAddChildGraphNode':[C.POINTER(P),P,C.POINTER(P),S,P],
          'cuGraphInstantiateWithFlags':[C.POINTER(P),P,H],
          'cuGraphLaunch':[P,P],'cuGraphDestroy':[P],'cuGraphExecDestroy':[P],
          'cuGetErrorName':[C.c_int,C.POINTER(C.c_char_p)],
        }
        for name,args in bindings.items():
            f=getattr(self.driver,name);f.argtypes=args;f.restype=C.c_int
        bindings={
          'nvrtcVersion':[C.POINTER(C.c_int),C.POINTER(C.c_int)],
          'nvrtcCreateProgram':[C.POINTER(P),C.c_char_p,C.c_char_p,C.c_int,C.POINTER(C.c_char_p),C.POINTER(C.c_char_p)],
          'nvrtcCompileProgram':[P,C.c_int,C.POINTER(C.c_char_p)],
          'nvrtcGetProgramLogSize':[P,C.POINTER(S)],'nvrtcGetProgramLog':[P,C.c_char_p],
          'nvrtcGetPTXSize':[P,C.POINTER(S)],'nvrtcGetPTX':[P,C.c_char_p],
          'nvrtcDestroyProgram':[C.POINTER(P)],
        }
        for name,args in bindings.items():
            f=getattr(self.nvrtc,name);f.argtypes=args;f.restype=C.c_int

    def call(self,name,*args):
        rc=getattr(self.driver,name)(*args)
        if rc:
            text=C.c_char_p();self.driver.cuGetErrorName(rc,C.byref(text))
            raise RuntimeError(f'{name}: {rc} {text.value}')
    def rtc(self,name,*args):
        rc=getattr(self.nvrtc,name)(*args)
        if rc:raise RuntimeError(f'{name}: {rc}')
    def graph(self):
        g=C.c_void_p();self.call('cuGraphCreate',C.byref(g),0);return g
    def handle(self,g):
        h=C.c_ulonglong();self.call('cuGraphConditionalHandleCreate',C.byref(h),g,self.context,1,1);return h
    def dependencies(self,nodes):return (C.c_void_p*len(nodes))(*[n.value for n in nodes]) if nodes else None
    def kernel(self,g,name,values,deps=()):
        pointers=(C.c_void_p*len(values))(*[C.cast(C.byref(v),C.c_void_p).value for v in values])
        p=KernelParams(self.functions[name],1,1,1,1,1,1,0,pointers,None)
        n=C.c_void_p();self.call('cuGraphAddKernelNode',C.byref(n),g,self.dependencies(deps),len(deps),C.byref(p));return n
    def conditional(self,g,h,deps=()):
        p=NodeParams();p.type=13;p.params.conditional=Conditional(h.value,1,1,None,self.context)
        n=C.c_void_p();self.call('cuGraphAddNode',C.byref(n),g,self.dependencies(deps),len(deps),C.byref(p))
        return C.c_void_p(p.params.conditional.graphs[0])
    def instantiate(self,g):
        e=C.c_void_p();self.call('cuGraphInstantiateWithFlags',C.byref(e),g,0);return e
    def launch(self,e):self.call('cuGraphLaunch',e,C.c_void_p(torch.cuda.current_stream().cuda_stream))
    def child(self,g,raw,deps=()):
        n=C.c_void_p();self.call('cuGraphAddChildGraphNode',C.byref(n),g,self.dependencies(deps),len(deps),C.c_void_p(raw));return n
    def loop(self,first,body,done,count,lanes):
        g=self.graph();h=self.handle(g)
        start=self.child(g,first.raw_cuda_graph())
        args=[h,C.c_ulonglong(count.data_ptr()),C.c_ulonglong(done.data_ptr()),C.c_int(lanes),C.c_int(0)]
        gate=self.kernel(g,'condition',args,(start,))
        inside=self.conditional(g,h,(gate,))
        step=self.child(inside,body.raw_cuda_graph())
        args[-1]=C.c_int(1)
        self.kernel(inside,'condition',args,(step,))
        try:e=self.instantiate(g)
        except BaseException:self.call('cuGraphDestroy',g);raise
        return g,e

    def close(self):
        if getattr(self,"module",None) is not None:
            with torch.cuda.device(self.device):self.call("cuModuleUnload",self.module)
            self.module=None
    def __del__(self):
        try:self.close()
        except Exception:pass  # Interpreter shutdown; explicit close still reports errors.
