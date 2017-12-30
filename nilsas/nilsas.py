from __future__ import division
import os
import sys
import numpy as np
from scipy import sparse
import scipy.sparse.linalg as splinalg
from copy import deepcopy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .forward import Forward
from .segment import Segment
from .interface import Interface, adjoint_terminal_condition

def vector_bundle(
        run_forward, run_adjoint, u0, parameter, M_modes, K_segment, 
        nstep_per_segment, runup_steps, dt, stepfwfunc, stepsegfunc):

    # run_forward is a function in the form
    # inputs  - u0:     shape(m,). init solution.
    #           steps:  number of time steps, an int.
    # outputs - u:      shape (nstep, m), Trajectory 
    #           f:      shape (nstep, m). du/dt
    #           fu:     shape (nstep, m, m). Jacobian matrices 
    #           fs:     shape (nstep, m). pf/ps
    #           J:      shape (nstep,).
    #           Ju:     shape (nstep, m). pJ/pu
    #           Js:     shape (nstep, m). pJ/ps
    # Here m is the dimension of the dynamical system
    #
    # run_adjoint is a function in the form:
    # inputs -  w_tmn:      shape (M_modes, m). 
    #                       Terminal conditions of homogeneous adjoint
    #           yst_tmn:    shape (m,). Terminal condition of y^*_i
    #           vst_tmn:    shape (m,). Terminal condition of v^*_i
    #           fu:         shape (nstep, m, m). Jacobian
    #           Ju:         shape (nstep, m). partial J/ partial u,
    # outputs - w:          shape (nstep, M_modes, m).
    #                       homogeneous solutions on the segment
    #           yst:        shape (nstep, m). y^* for genereating neutral CLV
    #           vst:        shape (nstep, m). inhomogeneous solution
    
    forward = Forward()
    forward.run(run_forward, u0, parameter, nstep_per_segment, 
            K_segment,  runup_steps, dt, stepfwfunc)
    assert not np.isnan(forward.f[-1,-1,0]), \
            'f=du/dt is becoming nan.'
    assert not np.allclose(np.linalg.norm(forward.f[-1,-1,0]), 0, \
            atol=1e-11), 'f=du/dt is becoming zero.'

    segment = Segment()
    interface = Interface()
    interface.terminal(M_modes, forward)

    for i in range(K_segment-1, -1, -1):
        segment.run1seg(run_adjoint, interface, forward, dt, stepsegfunc)
        interface.interface_right(segment)
        interface.rescale(forward.f[i,0])

    return forward, interface, segment


def nilsas_min(C, R, d, b):
    # solves the minimization problem for y or v, obtain coefficients a
    K_segment, M_modes = d.shape
    assert type(C) == type(R) == type(d) == type(b) == np.ndarray
    assert C.ndim == 3 and R.ndim == 3 and d.ndim == 2 and b.ndim == 2
    assert C.shape[0] == R.shape[0]-1 == d.shape[0] \
            == b.shape[0]-1 == K_segment
    assert C.shape[1] == C.shape[2] == R.shape[1] == R.shape[2] \
            == d.shape[1] == b.shape[1] == M_modes

    R = R[1:-1] # do not need first and last interface
    b = b[1:-1]

    eyes = np.eye(M_modes, M_modes) * np.ones([K_segment-1, 1, 1])
    B_shape = (M_modes * (K_segment-1), M_modes * K_segment)
    I = sparse.bsr_matrix(
            (eyes, np.r_[:K_segment-1], np.r_[:K_segment]), shape=B_shape)
    D = sparse.bsr_matrix(
            (R, np.r_[1:K_segment], np.r_[:K_segment]))
    B = (I - D).tocsr()

    Cinv = np.array([np.linalg.inv(C[i]) for i in range(K_segment)])
    C = sparse.bsr_matrix((C, np.r_[:K_segment], np.r_[:K_segment+1]))
    Cinv = sparse.bsr_matrix((Cinv, np.r_[:K_segment], np.r_[:K_segment+1]))
    C = C.tocsr()
    Cinv = Cinv.tocsr()

    d = np.ravel(d)
    b = np.ravel(b)

    Schur = B * Cinv * B.T 
    lbd = - splinalg.spsolve(Schur, B*Cinv*d + b)
    a = - Cinv * (B.T * lbd + d) 
    assert len(a) == K_segment * M_modes
    return a.reshape([K_segment, M_modes]), C, Cinv, B


def gradient(forward, segment):
    # computes the gradient from checkpoints
    # inputs -  segment_range: the checkpoints to be used for sensitivity
    K_segment, nstep_per_segment, _ = forward.f.shape
    grad = (segment.v[:,:,np.newaxis,:] * forward.fs).sum((0,1,3)) \
            / K_segment / nstep_per_segment
    Javg = np.average(forward.J)
    return Javg, grad


def nilsas_main(
        run_forward, run_adjoint, u0, parameter, M_modes, K_segment, 
        nstep_per_segment, runup_steps, dt, stepfwfunc, stepsegfunc):
    
    # get vector bundles
    forward, interface, segment =  vector_bundle( 
            run_forward, run_adjoint, u0, parameter, M_modes, 
            K_segment, nstep_per_segment, runup_steps, dt,
            stepfwfunc, stepsegfunc)
 
    # solve nilsas problem for y, then compute y, v^pm, d^v
    ay, _, _, _ = nilsas_min(segment.C, interface.R, 
            segment.dy, interface.by )
    segment.y_vstpm_dv(ay, forward.f)

    # solve nilsas problem for v
    av, _, _, _ = nilsas_min(segment.C, interface.R, 
            segment.dv, interface.bv)
    segment.vpm_v(av, forward.f, forward.Jtild)

    # compute gradient
    Javg, grad = gradient(forward, segment)

    return Javg, grad, forward, interface, segment
