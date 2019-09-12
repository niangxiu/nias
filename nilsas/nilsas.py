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
from pdb import set_trace

from .forward import Forward
from .segment import Segment
from .interface import Interface, adjoint_terminal_condition

def vector_bundle(u0, M_modes, K_segment, 
        nstep_per_segment, runup_steps, stepfwfunc, stepsegfunc, derivatives, dudt):

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
    
    forward = Forward()
    forward.run_allseg(u0, nstep_per_segment, K_segment,  runup_steps, stepfwfunc, derivatives, dudt)
    assert not np.isnan(forward.f[-1,-1, 0]), 'f=du/dt is becoming nan.'
    assert not np.allclose(np.linalg.norm(forward.f[-1,-1]), 0, \
            atol=1e-11), 'f=du/dt is becoming zero.'

    segment = Segment()
    interface = Interface()
    interface.terminal(M_modes, forward)

    for i in range(K_segment-1, -1, -1):
        segment.run1seg(interface, forward, stepsegfunc)
        interface.interface_right(segment)
        interface.rescale(forward.f[i,0])

    return forward, interface, segment


def nilsas_min(C, R, dwv, dwf, dvf, b):
    # solves the minimization problem for y or v, obtain coefficients a
    K_segment, M_modes = dwv.shape
    assert type(C) == type(R) == type(dwv) == type(b)\
            == type(dwf) == type(dvf) == np.ndarray
    assert C.ndim == R.ndim == 3 \
            and dwv.ndim == dwf.ndim == b.ndim == 2
    assert C.shape[0] == R.shape[0]-1 == dwv.shape[0] == dwf.shape[0] \
            == dvf.shape[0] == b.shape[0]-1 == K_segment
    assert C.shape[1] == C.shape[2] == R.shape[1] == R.shape[2] \
            == dwv.shape[1] == dwf.shape[1] == b.shape[1] == M_modes
    assert dvf.shape == (K_segment,)

    R = R[1:-1] # do not need first and last interface
    b = b[1:-1]

    eyes = np.eye(M_modes, M_modes) * np.ones([K_segment-1, 1, 1])
    B_shape = (M_modes * (K_segment-1), M_modes * K_segment)
    I = sparse.bsr_matrix(
            (eyes, np.r_[:K_segment-1], np.r_[:K_segment]), shape=B_shape)
    D = sparse.bsr_matrix(
            (R, np.r_[1:K_segment], np.r_[:K_segment]))
    # B = (I - D).tocsr()
    B = sparse.vstack([(I-D), [dwf.ravel()]]).tocsr()

    Cinv = np.array([np.linalg.inv(C[i]) for i in range(K_segment)])
    C = sparse.bsr_matrix((C, np.r_[:K_segment], np.r_[:K_segment+1]))
    Cinv = sparse.bsr_matrix((Cinv, np.r_[:K_segment], np.r_[:K_segment+1]))
    C = C.tocsr()
    Cinv = Cinv.tocsr()

    d = dwv.ravel()
    b = np.append(b.ravel(), -dvf.sum())

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
    grad += forward.Js.sum((0,1)) / K_segment / nstep_per_segment
    Javg = np.average(forward.J)
    return Javg, grad


def nilsas_main(u0, M_modes, K_segment, 
        nstep_per_segment, runup_steps, stepfwfunc, stepsegfunc, derivatives, dudt):
    
    # get vector bundles
    forward, interface, segment =  vector_bundle( 
            u0, M_modes, K_segment, nstep_per_segment, runup_steps,
            stepfwfunc, stepsegfunc, derivatives, dudt)
 
    # solve nilsas problem for v
    av, _, _, _ = nilsas_min(segment.C, interface.R, segment.dwv,
            segment.dwf, segment.dvf, interface.bv)
    segment.get_v(av, forward.f, forward.Jtild)

    # compute gradient
    Javg, grad = gradient(forward, segment)

    return Javg, grad, forward, interface, segment
