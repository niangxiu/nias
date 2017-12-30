from __future__ import division
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shutil
import sys

sys.path.append("../")
from nilsas.nilsas import nilsas_main

# this application has two paramters: rho, sigma
# the base for rho is 30
# the base for sigma is 10
# the largest time step allowed for PTA forward Euler: dt = 0.001
beta = 8/3.0

def derivatives(u, rho, sigma):
    [x, y, z] = u
    J   = z
    fu  = np.array([[-sigma, sigma, 0], [rho - z, -1, -x], [y, x, -beta]])
    Ju  = np.array([0, 0, 1])
    fs  = np.array([[0, x, 0], [y-x, 0, 0]])
    return J, fu, Ju, fs


def step_PTA(u, f, fu, rho, dt, sigma):
    u_next = u + f * dt
    f_next = f + np.dot(fu, f) * dt
    return u_next, f_next


def step_PTA_backward_Euler(u, f, fu, rho, dt, sigma):
    f_next = np.linalg.solve(np.eye(3) - fu*dt, f) 
    u_next = u + f*dt
    return u_next, f_next


def dudt(u, rho, sigma):
    [x, y, z] = u
    f = np.array([sigma*(y-x), x*(rho-z)-y, x*y-beta*z])
    return f


def step_forward_euler(u, f, fu, rho, dt, sigma):
    u_next = u + f * dt
    return u_next, dudt(u_next, rho, sigma)


def step_RK4(u, f, fu, rho, dt, sigma):
    k0 = dt * dudt(u, rho, sigma) 
    k1 = dt * dudt(u + 0.5 * k0, rho, sigma)
    k2 = dt * dudt(u + 0.5 * k1, rho, sigma)
    k3 = dt * dudt(u + k2, rho, sigma)
    u_next = u + (k0 + 2*k1 + 2*k2 + k3) / 6.0
    return u_next, dudt(u_next, rho, sigma)


def run_forward(u0, base_parameter, nstep, dt, stepfunc, f0=None):
    # run_forward is a function in the form
    # inputs  - u0:     shape (m,). initial state
    #           nstep:  scalar. number of time steps.
    #           base_parameter: tuple (rho, sigma). 
    # outputs - u: shape (nstep, m). m is dymension of  system. Trajectory.
    #           f: shape (nstep, m). du/dt
    #           fu: shape (nstep, m, m). Jacobian matrices
    #           fs: shape (nstep, ns, m). pf/ps
    #           J: shape (nstep,).
    #           Ju: shape (nstep, m). pJ/pu
    #           Js: shape (nstep, ns, m). pJ/ps

    rho, sigma = base_parameter
    m   = 3
    ns  = 2 # number of parameters

    assert len(u0) == m

    u   = np.zeros([nstep+1, m])
    f   = np.zeros([nstep+1, m])
    fu  = np.zeros([nstep+1, m, m])
    fs  = np.zeros([nstep+1, ns, m])
    J   = np.zeros([nstep+1])
    Ju  = np.zeros([nstep+1, m])
    Js  = np.zeros([nstep+1, ns, m])
    
    # zeroth step value save
    [x,y,z] = u0
    if f0 is not None:
        assert len(f0) == m
    else:
        f0  = np.array([sigma*(y-x), x*(rho-z)-y, x*y-beta*z])
    J_, fu_, Ju_, fs_ = derivatives(u0, rho, sigma)
    u[0]    = u0
    f[0]    = f0
    fu[0]   = fu_
    fs[0]   = fs_
    J[0]    = J_
    Ju[0]   = Ju_

    for i in range(1, 1+nstep):
        u_next, f_next = stepfunc(u[i-1], f[i-1], fu[i-1], rho, dt, sigma)
        J_, fu_, Ju_, fs_ = derivatives(u_next, rho, sigma)
        u[i]    = u_next
        f[i]    = f_next
        fu[i]   = fu_
        fs[i]   = fs_
        J[i]    = J_
        Ju[i]   = Ju_

    return u, f, fu, fs, J, Ju, Js


def adjoint_step_explicit(fu, Ju, adjall, dt):
    adjall_next = (np.dot(fu.T, adjall.T) * dt + adjall.T).T
    adjall_next[-1] += Ju * dt
    return adjall_next


def adjoint_step_implicit(fu, Ju, adjall, dt):
    adjall[-1] += Ju * dt
    adjall_next = (np.linalg.solve(np.eye(3)-fu.T*dt, adjall.T)).T
    return adjall_next


def run_adjoint(w_tmn, yst_tmn, vst_tmn, fu, Ju, dt, stepfunc):
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
        adjall_next = stepfunc(fu[i], Ju[i], adjall, dt)
        w.insert(0,adjall_next[:-2])
        yst.insert(0, adjall_next[-2])
        vst.insert(0, adjall_next[-1])
        adjall = adjall_next

    w   = np.array(w)
    yst = np.array(yst)
    vst = np.array(vst)

    return w, yst, vst


if __name__ == '__main__': # pragma: no cover

    # parameters (rho, sigma)
    M_modes = 1
    nstep_per_segment = 1000
    runup_steps = 10000
    dt = 0.01
    K_segment = 2
    rho = 28
    sigma = 10
    u0 = [0,1,2]
    parameter = (rho, sigma)
    n_repeat = 2

    Javg_   = []
    grad_   = []

    # trial run
    # Javg, grad, forward, interface, segment = nilsas_main(
            # run_forward, run_adjoint, u0, parameter, M_modes,
            # K_segment, nstep_per_segment, runup_steps, dt, 
            # step_RK4, adjoint_step_explicit)
    
    # plt.plot((forward.f*segment.y).sum(-1).flat)
    # plt.savefig('fy.png')
    # plt.close()

    # plot trajectory
    # fig = plt.figure()
    # plt.plot(, Javg_, '.')
    # plt.savefig('rho_J.png')
    # plt.close(fig)

    # plot different rho
    # rho_    = []
    # for rho in np.arange (25, 50.1, 0.5):
        # parameter   = (rho, sigma)
        # Javg, grad = nilsas_main(
            # run_forward, run_adjoint, u0, parameter, M_modes, K_segment, 
            # nstep_per_segment, runup_steps, dt, stepfunc=step_forward_euler)
        # print(K_segment, Javg, grad, '    ', u0)
        # rho_.append(rho)
        # Javg_.append(Javg) 
        # grad_.append(grad)

    # Javg_ = np.array(Javg_)
    # grad_ = np.array(grad_)

    # fig = plt.figure()
    # plt.plot(rho_, Javg_, '.')
    # plt.savefig('rho_J.png')
    # plt.close(fig)

    # fig = plt.figure()
    # plt.plot(rho_, grad_[:,0], '.')
    # plt.ylim(0, 2)
    # plt.savefig('rho_grad.png')
    # plt.close(fig)

    # plot different trajectory length
    # K_segment_ = np.array([100], dtype=int) #, 1e3, 2e3, 5e3, 1e4, 2e4])
    # T_ = K_segment_ * dt * nstep_per_segment
    # for K_segment, T in zip(K_segment_, T_):
        # Javg__ = []
        # grad__ = []
        # for _ in range(n_repeat):
            # u0 = np.random.rand(3) * 20
            # Javg, grad, forward, interface, segment = nilsas_main(
                # run_forward, run_adjoint, u0, parameter, M_modes,
                # K_segment, nstep_per_segment, runup_steps, dt, 
                # step_RK4, adjoint_step_explicit)
            # print(K_segment, Javg, grad)
            # Javg__.append(Javg)
            # grad__.append(grad)
        # Javg_.append(np.array(Javg__)) 
        # grad_.append(np.array(grad__))
    # Javg_ = np.array(Javg_)
    # grad_ = np.array(grad_)

    # fig = plt.figure()
    # plt.semilogx(T_, Javg_, '.')
    # plt.savefig('T_J.png')
    # plt.close(fig)

    # fig = plt.figure()
    # plt.semilogx(T_, grad_[:,:,0], '.')
    # plt.savefig('T_grad.png')
    # plt.close(fig)


