from __future__ import division
import os
import sys
import numpy as np
from copy import deepcopy
from .forward import Forward
from .segment import Segment
from .interface import Interface, adjoint_terminal_condition

def vector_bundle(
        run_forward, run_adjoint, u0, parameter, M_modes, K_segment, 
        nstep_per_segment, runup_steps, dt):

    # run_forward is a function in the form
    # inputs  - u0:     shape(m,). init solution.
    #           steps:  number of time steps, an int.
    # outputs - u:      shape (nstep, m), where m is dymension of dynamical system. Trajectory 
    #           f:      shape (nstep, m). du/dt
    #           fu:     shape (nstep, m, m). Jacobian matrices 
    #           fs:     shape (nstep, m). pf/ps
    #           J:      shape (nstep,).
    #           Ju:     shape (nstep, m). pJ/pu
    #           Js:     shape (nstep, m). pJ/ps
    #
    # run_adjoint is a function in the form:
    # inputs -  w_tmn:      shape (M_modes, m). Terminal conditions of homogeneous adjoint
    #           yst_tmn:    shape (m,). Terminal condition of y^*_i
    #           vst_tmn:    shape (m,). Terminal condition of v^*_i
    #           fu:         shape (nstep, m, m). Jacobian
    #           Ju:         shape (nstep, m). partial J/ partial u,
    # outputs - w:          shape (nstep, M_modes, m). homogeneous solutions on the segment
    #           yst:        shape (nstep, m). y^*, for genereating neutral CLV
    #           vst:        shape (nstep, m). inhomogeneous solution
    
    forward = Forward()
    forward.run(run_forward, u0, parameter, nstep_per_segment, K_segment,  runup_steps, dt)

    segment = Segment()
    interface = Interface()
    interface.terminal(M_modes, forward)

    for i in range(K_segment-1, -1, -1):
        segment.run1seg(run_adjoint, interface, forward, dt)
        interface.interface_right(segment)
        interface.rescale(forward.f[i,0])

    return forward, interface, segment


def nilsas_min(C, R, d, b):
    # solves the minimization problem for y or v, obtain coefficients a

    K_segment, M_modes = d.shape
    assert type(C) == type(R) == type(d) == type(b) == np.ndarray
    assert C.ndim == 3 and R.ndim == 3 and d.ndim == 2 and b.ndim == 2
    assert C.shape[0] == R.shape[0] == d.shape[0] == b.shape[0] == K_segment
    assert C.shape[1] == c.shape[2] == R.shape[1] == R.shape[2] == d.shape[1] == b.shape[1] == M_modes

    eyes    = np.eye(M_modes, M_modes) * np.ones([K_segment-1, 1, 1])
    B_shape = (M_modes * (K_segment-1), M_modes * K_segment)
    I       = sparse.bsr_matrix((eyes,  np.r_[:K_segment-1],    np.r_[:K_segment]), shape=B_shape)
    D       = sparse.bsr_matrix((R,     np.r_[1:K_segment],     np.r_[:K_segment]))
    B       = (I - D).tocsr()

    Cinv    = np.array([np.linalg.inv(C[i]) for i in range(K_segment)])
    C       = sparse.bsr_matrix((C,     np.r_[:K_segment],  np.r_[:K_segment+1]))
    Cinv    = sparse.bsr_matrix((Cinv,  np.r_[:K_segment],  np.r_[:K_segment+1]))

    d       = np.ravel(d)
    b       = np.ravel(b)

    Schur   = B * Cinv * B.T 
    lbd     = - splinalg.spsolve(Schur, B*Cinv*d + b)
    a       = - Cinv * (B.T * lbd + d) 
    assert len(a) == K_segment * M_modes
    return a.reshape([K_segment, M_modes])


def gradient(checkpoint, segment_range=None):
    # computes the gradient from checkpoints
    # inputs -  segment_range: the checkpoints to be used for sensitivity
    # outputs - the sensitivity

    _, _, _, lss, G_lss, g_lss, J, G_dil, g_dil = checkpoint
    if segment_range is not None:
        lss = deepcopy(lss)
        if isinstance(segment_range, int):
            s = slice(segment_range)
        else:
            s = slice(*segment_range)
        lss.bs = lss.bs[s]
        lss.Rs = lss.Rs[s]
        G_lss  = G_lss [s]
        g_lss  = g_lss [s]
        J      = J     [s]
        G_dil  = G_dil [s]
        g_dil  = g_dil [s]
    alpha = lss.solve()
    grad_lss = (alpha[:,:,np.newaxis] * np.array(G_lss)).sum(1) + np.array(g_lss)
    J = np.array(J)
    dJ = trapez_mean(J.mean(0), 0) - J[:,-1]
    nstep_per_segment = J.shape[1]
    dil = ((alpha * G_dil).sum(1) + g_dil) / nstep_per_segment
    grad_dil = dil[:,np.newaxis] * dJ
    return windowed_mean(grad_lss) + windowed_mean(grad_dil)


def nilsas_main():
    pass
