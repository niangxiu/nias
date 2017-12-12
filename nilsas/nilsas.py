from __future__ import division
import os
import sys
import numpy as np
from copy import deepcopy

# from .checkpoint import Checkpoint, verify_checkpoint, save_checkpoint
# from .timedilation import TimeDilation, TimeDilationExact
# from .lsstan import LssTangent#, tangent_initial_condition
from .timeseries import windowed_mean
from .utility import qr_transpose

def adjoint_terminal_condition(M_modes, f_tmn):
    # inputs -  M_modes:    number of homogeneous adjoint
    #           m:          the dimension of the dynamical system
    #           f_tmn:      f at the end of the trajectory
    
    m = f_tmn.shape[0]
    assert M_modes <= m, "number of modes should be smaller than dimension of system"

    W_  = np.random.rand(M_modes, m)
    W__ = W_ - np.dot(W_, f_tmn)[:,np.newaxis] * f_tmn / np.dot(f_tmn, f_tmn)
    W,_ = qr_transpose(W__)
    yst_tmn = f_tmn
    vst_tmn = np.zeros(m)

    return W, yst_tmn, vst_tmn


def nilsas_gradient(checkpoint, segment_range=None):
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
    steps_per_segment = J.shape[1]
    dil = ((alpha * G_dil).sum(1) + g_dil) / steps_per_segment
    grad_dil = dil[:,np.newaxis] * dJ
    return windowed_mean(grad_lss) + windowed_mean(grad_dil)


def continue_adj_shadowing(
        run, parameter, checkpoint, num_segments, steps_per_segment,
        checkpoint_path=None, checkpoint_interval=1):
    # the true output of this function is the checkpoint files saved at each interfaces 

    compute_outputs = []

    assert verify_checkpoint(checkpoint)
    u0, V, v, lss, G_lss, g_lss, J_hist, G_dil, g_dil = checkpoint

    manager = Manager()
    interprocess = (manager.Lock(), manager.dict())

    i = lss.K_segments()
    run_id = 'time_dilation_{0:02d}'.format(i)
    if run_ddt is not None:
        time_dil = TimeDilationExact(run_ddt, u0, parameter)
    else:
        time_dil = TimeDilation(run, u0, parameter, run_id,
                                simultaneous_runs, interprocess)

    V = time_dil.project(V)
    v = time_dil.project(v)
    V, v = lss.checkpoint(V, v)
    

    u0, V, v, J0, G, g = run_segment(
            run, u0, V, v, parameter, i, steps_per_segment,
            epsilon, simultaneous_runs, interprocess, get_host_dir=get_host_dir,
            compute_outputs=compute_outputs, spawn_compute_job=spawn_compute_job)

    J_hist.append(J0)
    G_lss.append(G)
    g_lss.append(g)

    for i in range(lss.K_segments() + 1, num_segments + 1):

        # time dilation contribution
        run_id = 'time_dilation_{0:02d}'.format(i)
        if run_ddt is not None:
            time_dil = TimeDilationExact(run_ddt, u0, parameter)
        else:
            time_dil = TimeDilation(run, u0, parameter, run_id,
                                    simultaneous_runs, interprocess)
        G_dil.append(time_dil.contribution(V))
        g_dil.append(time_dil.contribution(v))

        V = time_dil.project(V)
        v = time_dil.project(v)

        V, v = lss.checkpoint(V, v)
        # extra outputs to compute
        compute_outputs = [lss.Rs[-1], lss.bs[-1], G_dil[-1], g_dil[-1]]

        # run all segments
        if i < num_segments:
            u0, V, v, J0, G, g = run_segment(
                    run, u0, V, v, parameter, i, steps_per_segment,
                    epsilon, simultaneous_runs, interprocess, get_host_dir=get_host_dir,
                    compute_outputs=compute_outputs, spawn_compute_job=spawn_compute_job)
        else:
            run_compute(compute_outputs, spawn_compute_job=spawn_compute_job, interprocess=interprocess)

        for output in [lss.Rs, lss.bs, G_dil, g_dil]:
            output[-1] = output[-1].field

        checkpoint = Checkpoint(
                u0, V, v, lss, G_lss, g_lss, J_hist, G_dil, g_dil)
        print(lss_gradient(checkpoint))
        sys.stdout.flush()

        if checkpoint_path and (i) % checkpoint_interval == 0:
            save_checkpoint(checkpoint_path, checkpoint)

        if i < num_segments:
            J_hist.append(J0)
            G_lss.append(G)
            g_lss.append(g)

    G = lss_gradient(checkpoint)
    return np.array(J_hist).mean((0,1)), G


def adj_shadowing(
        run_primal, run_adjoint, u0, parameter, M_modes, num_segments, 
        steps_per_segment, runup_steps, checkpoint_path=None, checkpoint_interval=1):

    # run_primal is a function in the form
    # inputs  - u0:     init solution, a flat numpy array of doubles.
    #           steps:  number of time steps, an int.
    # outputs - u_end:  final solution, a flat numpy array of doubles, must be of the same size as u0.
    #           J:      quantities of interest, a numpy array of shape (steps,)
    #           Df:     Jacobian, shape (m, m, steps), where m is dimension of dynamical system
    #           Ju:     partial J/ partial u, shape (m, steps)
    #
    # run_adjoint is a function in the form:
    # inputs -  w_tmn:      terminal condition of homogeneous adjoint, of shape (M_modes, m)
    #           yst_tmn:    terminal condition of y^*, of shape (m,)
    #           vst_tmn:    terminal condition of v^*, of shape (m,)
    #           Df:         Jacobian, shape (m, m, steps), where m is dimension of dynamical system
    #           Ju:         partial J/ partial u, shape (m, steps)
    # outputs - w_bgn:      homogeneous solutions at the beginning of the segment, of shape (M_modes, m)
    #           yst_bgn:    ystar, for genereating neutral CLV, of shape (m,)
    #           vst_bgn:    inhomogeneous solution, of shape (m,)
    
    if runup_steps > 0:
        u0, f0, _, _, _, _ = run_primal(u0, runup_steps)

    W, yst_tmn, vst_tmn = adjoint_terminal_condition(M_modes, f0[-1])
    lss = LssTangent()
    checkpoint = Checkpoint(u0, V, v, lss, [], [], [], [], [])
    return continue_shadowing(
            run, parameter, checkpoint,
            num_segments, steps_per_segment, epsilon,
            checkpoint_path, checkpoint_interval,
            simultaneous_runs, run_ddt, return_checkpoint, get_host_dir, spawn_compute_job)

