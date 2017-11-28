import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shutil
import sys

sys.path.append("../nilsas/")
from nilsas import *

def ddt(uwvs):
    u = uwvs[0]
    [x, y, z] = u
    w = uwvs[1]
    vstar = uwvs[2]
    dudt = np.array([sigma * (y - x), x * (rho - z) - y, x * y - beta * z])
    Df = np.array([[-sigma, sigma, 0],[rho - z,-1,-x],[y,x,-beta]])
    dwdt = np.dot(Df, w.T)
    dfdrho = np.array([0, x, 0])
    dvstardt = np.dot(Df, vstar) + dfdrho
    return np.array([dudt, dwdt.T, dvstardt])

def RK1(u, w, vstar):
    # integrate u, w, and vstar to the next time step
    uwvs = np.array([u, w, vstar])
    k0 = dt * ddt(uwvs) 
    uwvs_new = uwvs + k0 
    return uwvs_new

def RK4(u, w, vstar):
    # integrate u, w, and vstar to the next time step
    uwvs = np.array([u, w, vstar])
    k0 = dt * ddt(uwvs) 
    k1 = dt * ddt(uwvs + 0.5 * k0)
    k2 = dt * ddt(uwvs + 0.5 * k1)
    k3 = dt * ddt(uwvs + k2)
    uwvs_new = uwvs + (k0 + 2*k1 + 2*k2 + k3) / 6.0
    return uwvs_new

def solve_primal(u0, steps):
    # a function in the form
    # inputs  - u0:     init solution, a flat numpy array of doubles.
    #           steps:  number of time steps, an int.
    # outputs - u_end:  final solution, a flat numpy array of doubles, must be of the same size as u0.
    #           J:      quantities of interest, a numpy array of shape (steps,)
    #           Df:     Jacobian, shape (m, m, steps), where m is dimension of dynamical system
    #           Ju:     partial J/ partial u, shape (m, steps)
    
    return u_end, J, Df, Ju

def solve_adjoint(w_tmn, vih_tmn, Df, fs):
    # inputs -  w_tmn:  terminal condition of homogeneous adjoint, of shape (M_modes, m)
    #           vih_tmn:terminal condition of inhomogeneous adjoint, of shape (m,)
    #           Df:     Jacobian, shape (m, m, steps), where m is dimension of dynamical system
    #           Ju:     partial J/ partial u, shape (m, steps)
    # outputs - w:      homogeneous solutions on this segment, of shape (M_modes, m, steps)
    #           vih:    inhomogeneous solution on this segment, of shape (m, steps)

    return w, vih
