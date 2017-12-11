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


def solve_primal(u0, steps):
    # a function in the form
    # inputs  - u0:         init solution, a flat numpy array of doubles.
    #           steps:      number of time steps, an int.
    # outputs - u_end:      final solution, a flat numpy array of doubles, must be of the same size as u0.
    #           J:          quantities of interest, a numpy array of shape (steps,)
    #           fu:         Jacobian, shape (m, m, steps), where m is dimension of dynamical system
    #           Ju:         partial J/ partial u, shape (m, steps)
    
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

    for i in range(steps):
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

    # plot for debug
    plt.plot(u[:,0], u[:,2])
    plt.savefig('lorenz.png')
    plt.close()
    
    return u, f, J, fu, Ju, fs


def solve_adjoint(w_tmn, vih_tmn, Df, fs):
    # inputs -  w_tmn:      terminal condition of homogeneous adjoint, of shape (M_modes, m)
    #           yst_tmn:    terminal condition of y^*, of shape (m,)
    #           vst_tmn:    terminal condition of v^*, of shape (m,)
    #           Df:         Jacobian, shape (m, m, steps), where m is dimension of dynamical system
    #           Ju:         partial J/ partial u, shape (m, steps)
    # outputs - w_bgn:      homogeneous solutions at the beginning of the segment, of shape (M_modes, m)
    #           yst_bgn:    y^*, for genereating neutral CLV, of shape (m,)
    #           vst_bgn:    inhomogeneous solution, of shape (m,)

    return w, yst_bgn, vst_bgn


# the main function, now serves as test

# from nilsas.utility import qr_transpose
# A = np.array(np.random.rand(4,6))
# Q, R = qr_transpose(A)

# from nilsas.nilsas import adjoint_terminal_condition
# W, vih = adjoint_terminal_condition(2,3)

solve_primal([0,1,5], 10000)
