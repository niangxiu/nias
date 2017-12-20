from __future__ import division
import os
import sys
import numpy as np
from copy import deepcopy
from forward import Forward
from segment import Segment
from interface import Interface, adjoint_terminal_condition

def vector_bundle(
        run_forward, run_adjoint, u0, parameter, M_modes, K_segments, 
        nstep_per_segment, runup_steps=0, checkpoint_path=None):

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
    forward.run(run_forward, u0, nstep_per_segment, K_segments, runup_steps)

    segment = Segment()
    interface = Interface()
    interface.terminal_right(M_modes, forward)
    interface.rescale()
    for i in range(K_segments):
        segment.run1seg(run_adjoint, interface, forward)
        interface.interface_right(segment)
        interface.rescale()

    return forward, interface, segment


def nilsas_min:
    # solves the minimization problem for y or v,
    # obtain a_i 
    pass


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


def nilsas_main:
    pass
