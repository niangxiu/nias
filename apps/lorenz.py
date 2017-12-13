from __future__ import division
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shutil
import sys

sys.path.append("../")
from nilsas import *

sigma = 10
beta = 8/3.0
dt = 0.001

def derivatives(u, rho):
    [x, y, z] = u
    # f   = np.array([sigma * (y - x), x * (rho - z) - y, x * y - beta * z])
    J   = z
    fu  = np.array([[-sigma, sigma, 0], [rho - z, -1, -x], [y, x, -beta]])
    Ju  = np.array([0, 0, 1])
    fs  = np.array([0, x, 0])
    return J, fu, Ju, fs


def stepPTA(u, f, fu, rho):
    u_next = u + f * dt
    f_next = f + np.dot(fu, f) * dt
    return u_next, f_next


def solve_primal(u0, nsteps):
    # a function in the form
    # inputs  - u0:         init solution, a flat numpy array of doubles.
    #           nsteps:     number of time steps, an int.
    # outputs - u_end:      final solution, a flat numpy array of doubles, must be of the same size as u0.
    #           J:          quantities of interest, a numpy array of shape (nsteps,)
    #           fu:         Jacobian, shape (m, m, nsteps), where m is dimension of dynamical system
    #           Ju:         partial J/ partial u, shape (m, nsteps)
    
    rho = 30
    u   = []
    f   = []
    J   = []
    fu  = []
    Ju  = []
    fs  = []
    
    # zeroth step value save
    [x,y,z] = u0
    f0  = np.array([sigma * (y - x), x * (rho - z) - y, x * y - beta * z])
    J_, fu_, Ju_, fs_ = derivatives(u0, rho)
    u.append(u0)
    f.append(f0)
    J.append(J_)
    fu.append(fu_)
    Ju.append(Ju_)
    fs.append(fs_)

    for i in range(nsteps):
        u_next, f_next = stepPTA(u[-1], f[-1], fu[-1], rho)
        J_, fu_, Ju_, fs_ = derivatives(u_next, rho)
        u.append(u_next)
        f.append(f_next)
        J.append(J_)
        fu.append(fu_)
        Ju.append(Ju_)
        fs.append(fs_)

    u   = np.array(u)
    f   = np.array(f)
    J   = np.array(J)
    fu  = np.array(fu)
    Ju  = np.array(Ju)
    fs  = np.array(fs)

    return u, f, J, fu, Ju, fs


def solve_adjoint(w_tmn, yst_tmn, vst_tmn, fu, Ju):
    # inputs -  w_tmn:      terminal condition of homogeneous adjoint, of shape (M_modes, m)
    #           yst_tmn:    terminal condition of y^*, of shape (m,)
    #           vst_tmn:    terminal condition of v^*, of shape (m,)
    #           Df:         Jacobian, shape (nsteps, m, m), where m is dimension of dynamical system
    #           Ju:         partial J/ partial u, shape (nsteps, m)
    # outputs - w:          homogeneous solutions at the beginning of the segment, of shape (nsteps, M_modes, m)
    #           yst:        y^*, for genereating neutral CLV, of shape (nsteps, m)
    #           vst:        inhomogeneous solution, of shape (nsteps, m)

    nsteps = fu.shape[0] - 1
    M = w_tmn.shape[0]
    m = w_tmn.shape[1]

    w   = [w_tmn]
    yst = [yst_tmn]
    vst = [vst_tmn]
    adjall = np.vstack([w_tmn, yst_tmn, vst_tmn])
    
    for i in range(nsteps-1, -1, -1):
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


# the main function, now serves as test

# from nilsas.utility import qr_transpose
# A = np.array(np.random.rand(4,6))
# Q, R = qr_transpose(A)

