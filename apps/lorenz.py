from __future__ import division
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shutil
import sys

sys.path.append("../")
from nilsas import *

# this application has two paramters: rho, sigma
# the base for rho is 30
# the base for sigma is 10
beta = 8/3.0
dt = 0.001

def derivatives(u, rho, sigma):
    [x, y, z] = u
    # f   = np.array([sigma * (y - x), x * (rho - z) - y, x * y - beta * z])
    J   = z
    fu  = np.array([[-sigma, sigma, 0], [rho - z, -1, -x], [y, x, -beta]])
    Ju  = np.array([0, 0, 1])
    fs  = np.array([[0, x, 0], [y-x, 0, 0]])
    return J, fu, Ju, fs


def stepPTA(u, f, fu, rho):
    u_next = u + f * dt
    f_next = f + np.dot(fu, f) * dt
    return u_next, f_next


def run_forward(u0, base_parameter, nstep, f0=None):
    # run_forward is a function in the form
    # inputs  - u0:     shape (m,). initial state
    #           nstep:  scalar. number of time steps.
    #           base_parameter: tuple (rho, sigma). 
    # outputs - u:      shape (nstep, m), where m is dymension of dynamical system. Trajectory 
    #           f:      shape (nstep, m). du/dt
    #           fu:     shape (nstep, m, m). Jacobian matrices 
    #           fs:     shape (nstep, ns, m). pf/ps
    #           J:      shape (nstep,).
    #           Ju:     shape (nstep, m). pJ/pu
    #           Js:     shape (nstep, ns, m). pJ/ps

    rho, sigma = base_parameter
    # rho = 30
    # sigma = 10
    m   = 3
    ns  = 2 # number of parameters

    assert len(u0) == m
    if f0 is not None:
        assert len(f0) == m

    u   = np.zeros([nstep+1, m])
    f   = np.zeros([nstep+1, m])
    fu  = np.zeros([nstep+1, m, m])
    fs  = np.zeros([nstep+1, ns, m])
    J   = np.zeros([nstep+1])
    Ju  = np.zeros([nstep+1, m])
    Js  = np.zeros([nstep+1, ns, m])
    
    # zeroth step value save
    [x,y,z] = u0
    if f0 is None:
        f0  = np.array([sigma * (y - x), x * (rho - z) - y, x * y - beta * z])
    J_, fu_, Ju_, fs_ = derivatives(u0, rho, sigma)
    u[0]    = u0
    f[0]    = f0
    fu[0]   = fu_
    fs[0]   = fs_
    J[0]    = J_
    Ju[0]   = Ju_

    for i in range(1, 1+nstep):
        u_next, f_next = stepPTA(u[i-1], f[i-1], fu[i-1], rho)
        J_, fu_, Ju_, fs_ = derivatives(u_next, rho, sigma)
        u[i]    = u_next
        f[i]    = f_next
        fu[i]   = fu_
        fs[i]   = fs_
        J[i]    = J_
        Ju[i]   = Ju_

    return u, f, fu, fs, J, Ju, Js


def run_adjoint(w_tmn, yst_tmn, vst_tmn, fu, Ju):
    # inputs -  w_tmn:      shape (M_modes, m). terminal conditions of homogeneous adjoint
    #           yst_tmn:    shape (m,). terminal condition of y^*_i
    #           vst_tmn:    shape (m,). terminal condition of v^*_i
    #           fu:         shape (nstep, m, m). Jacobian
    #           Ju:         shape (nstep, m). partial J/ partial u,
    # outputs - w:          shape (nstep, M_modes, m). homogeneous solutions on the segment
    #           yst:        shape (nstep, m). y^*, for genereating neutral CLV
    #           vst:        shape (nstep, m). inhomogeneous solution

    nstep = fu.shape[0] - 1
    M = w_tmn.shape[0]
    m = w_tmn.shape[1]

    w   = [w_tmn]
    yst = [yst_tmn]
    vst = [vst_tmn]
    adjall = np.vstack([w_tmn, yst_tmn, vst_tmn])
    
    for i in range(nstep-1, -1, -1):
        adjall_next = (np.dot(fu[i].T, adjall.T) * dt + adjall.T).T
        adjall_next[-1] += Ju[i] * dt
        w.insert(0,adjall_next[:-2])
        yst.insert(0, adjall_next[-2])
        vst.insert(0, adjall_next[-1])
        adjall = adjall_next

    w   = np.array(w)
    yst = np.array(yst)
    vst = np.array(vst)

    return w, yst, vst
