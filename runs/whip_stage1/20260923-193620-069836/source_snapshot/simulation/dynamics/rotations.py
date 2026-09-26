"""Differentiable SO(3) helpers for rigid-body dynamics."""

import math
import torch


def skew(v):
    x,y,z=v.unbind(-1);o=torch.zeros_like(x)
    return torch.stack((o,-z,y,z,o,-x,-y,x,o),-1).reshape(*v.shape[:-1],3,3)


def rotation_exp(vector):
    angle=torch.linalg.vector_norm(vector,dim=-1)
    k=skew(vector)
    eye=torch.eye(3,dtype=vector.dtype,device=vector.device)
    a=torch.sinc(angle/math.pi)[...,None,None]
    b=(.5*torch.sinc(angle/(2*math.pi)).square())[...,None,None]
    return eye+a*k+b*(k@k)
