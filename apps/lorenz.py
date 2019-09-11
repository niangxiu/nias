from __future__ import division
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shutil
import sys
import pickle
import itertools
from multiprocessing import Pool, current_process
from pdb import set_trace

sys.path.append("../")
from nilsas.nilsas import nilsas_main

# this application has two paramters: rho, sigma
# the base for rho is 30 # the base for sigma is 10
# the largest time step allowed for PTA forward Euler: dt = 0.001
beta = 8/3.0
n_repeat = 4
dt = 0.001
nstep_per_segment = 200
M_modes = 2
K_segment = 20
runup_step = 2000
rho = 28
sigma = 10


def derivatives(u):
    [x, y, z] = u
    J   = z
    fu  = np.array([[-sigma, sigma, 0], [rho - z, -1, -x], [y, x, -beta]])
    Ju  = np.array([0, 0, 1])
    fs  = np.array([[0, x, 0], [y-x, 0, 0]])
    return J, fu, Ju, fs


def step_PTA(u, f, fu):
    u_next = u + f * dt
    f_next = f + np.dot(fu, f) * dt
    return u_next, f_next


def step_PTA_backward_Euler(u, f, fu):
    f_next = np.linalg.solve(np.eye(3) - fu*dt, f) 
    u_next = u + f*dt
    return u_next, f_next


def dudt(u):
    [x, y, z] = u
    f = np.array([sigma*(y-x), x*(rho-z)-y, x*y-beta*z])
    return f


def step_forward_euler(u, f, fu):
    u_next = u + f * dt
    return u_next, dudt(u_next)


def step_RK4(u, f, fu, rho, dt, sigma):
    k0 = dt * dudt(u, rho, sigma) 
    k1 = dt * dudt(u + 0.5 * k0, rho, sigma)
    k2 = dt * dudt(u + 0.5 * k1, rho, sigma)
    k3 = dt * dudt(u + k2, rho, sigma)
    u_next = u + (k0 + 2*k1 + 2*k2 + k3) / 6.0
    return u_next, dudt(u_next, rho, sigma)


def run_forward(u0, nstep, stepfunc, f0=None):
    """
    run_forward is a function in the form
    Args:
        u0:     shape (m,). initial state
        nstep:  scalar. number of time steps.
        base_parameter: tuple (rho, sigma). 
    Returns: 
        u: shape (nstep+1, m). m is dymension of  system. Trajectory.
        f: shape (nstep+1, m). du/dt
        fu: shape (nstep+1, m, m). Jacobian matrices
        fs: shape (nstep+1, ns, m). pf/ps
        J: shape (nstep+1,).
        Ju: shape (nstep+1, m). pJ/pu
        Js: shape (nstep+1, ns). pJ/ps
    """
    m = 3
    ns = 2 # number of parameters
    u = np.zeros([nstep+1, m])
    f = np.zeros([nstep+1, m])
    fu = np.zeros([nstep+1, m, m])
    fs = np.zeros([nstep+1, ns, m])
    J = np.zeros([nstep+1])
    Ju = np.zeros([nstep+1, m])
    Js = np.zeros([nstep+1, ns])
    
    # zeroth step value save
    assert len(u0) == m
    if f0 is not None:
        assert len(f0) == m
    else:
        f0  = dudt(u0)
    J_, fu_, Ju_, fs_ = derivatives(u0)
    u[0] = u0
    f[0] = f0
    fu[0] = fu_
    fs[0] = fs_
    J[0] = J_
    Ju[0] = Ju_

    for i in range(1, 1+nstep):
        u_next, f_next = stepfunc(u[i-1], f[i-1], fu[i-1])
        J_, fu_, Ju_, fs_ = derivatives(u_next)
        u[i]    = u_next
        f[i]    = f_next
        fu[i]   = fu_
        fs[i]   = fs_
        J[i]    = J_
        Ju[i]   = Ju_

    return u, f, fu, fs, J, Ju, Js


def adjoint_step_explicit(fu, Ju, adjall):
    adjall_next = (np.dot(fu.T, adjall.T) * dt + adjall.T).T
    adjall_next[-1] += Ju * dt
    return adjall_next


def adjoint_step_implicit(fu, Ju, adjall):
    adjall[-1] += Ju * dt
    adjall_next = (np.linalg.solve(np.eye(3)-fu.T*dt, adjall.T)).T
    return adjall_next


def run_adjoint(w_tmn, vst_tmn, fu, Ju, stepfunc):
    # inputs -  w_tmn:      shape (M_modes, m). terminal conditions of homogeneous adjoint
    #           vst_tmn:    shape (m,). terminal condition of v^*_i
    #           fu:         shape (nstep+1, m, m). Jacobian
    #           Ju:         shape (nstep+1, m). partial J/ partial u,
    # outputs - w:          shape (nstep+1, M_modes, m). homogeneous solutions on the segment
    #           vst:        shape (nstep+1, m). inhomogeneous solution

    nstep = fu.shape[0] - 1
    M = w_tmn.shape[0]
    m = w_tmn.shape[1]

    w   = [w_tmn]
    vst = [vst_tmn]
    adjall = np.vstack([w_tmn, vst_tmn])
    
    for i in range(nstep-1, -1, -1):
        adjall_next = stepfunc(fu[i], Ju[i], adjall)
        w.insert(0,adjall_next[:-1])
        vst.insert(0, adjall_next[-1])
        adjall = adjall_next

    w   = np.array(w)
    vst = np.array(vst)
    return w, vst


def u0_rand():
    u0 = np.zeros(3)
    for i in range(3):
        inrange = False
        while not inrange:
            u0[i] = (np.random.rand() - 0.5) * 30
            if abs(u0[i]) > 2:
                inrange = True
    return u0


def only_for_J_forwardEuler(u, nstep, runup_step):
    for _ in range(runup_step):
        [x, y, z] = u
        f = np.array([sigma*(y-x), x*(rho-z)-y, x*y-beta*z])
        u += f * dt
    J = np.zeros(nstep)
    for i in range(nstep):
        [x, y, z] = u
        f = np.array([sigma*(y-x), x*(rho-z)-y, x*y-beta*z])
        u += f * dt
        J[i] = u[-1]
    return J.mean()




def wrapped(i_repeat):
    u0 = u0_rand()
    Javg, grad, forward, interface, segment = nilsas_main(
        run_forward, run_adjoint, u0, M_modes,
        K_segment, nstep_per_segment, runup_step,
        step_forward_euler, adjoint_step_explicit)
    print(rho, sigma, Javg, grad)
    return [Javg, grad]


def all_info():
    # generate all info
    u0 = u0_rand()
    Javg, grad, forward, interface, segment = nilsas_main(
        run_forward, run_adjoint, u0, M_modes,
        K_segment, nstep_per_segment, runup_step, 
        step_forward_euler, adjoint_step_explicit)
    pickle.dump((Javg, grad, forward, interface, segment, dt, K_segment,
            nstep_per_segment),\
            open("all_info_segment.p", "wb"))


def change_rho():
    # grad for different rho
    global rho
    Javg_   = []
    grad_   = []
    rho_ = np.arange(25, 28, 1) 
    for rho in rho_:
        Javg__ = []
        grad__ = []
        for _ in range(n_repeat):
            u0 = u0_rand()
            Javg, grad, forward, interface, segment = nilsas_main(
                run_forward, run_adjoint, u0, parameter, M_modes,
                K_segment, nstep_per_segment, runup_step, 
                step_forward_euler, adjoint_step_explicit)
            print(rho, Javg, grad)
            Javg__.append(Javg)
            grad__.append(grad)
        Javg_.append(np.array(Javg__)) 
        grad_.append(np.array(grad__))
    Javg_ = np.array(Javg_)
    grad_ = np.array(grad_)
    pickle.dump((Javg_, grad_, rho_, sigma, M_modes, dt, \
            nstep_per_segment, K_segment, runup_step, n_repeat), \
            open("change_rho.p", "wb"))


def only_J():
    global rho
    Javg_   = []
    grad_   = []
    runup_step = 40000
    nstep = 100000
    n_repeat = 20
    rho_ = np.arange(24.75, 50.31, 0.25) 
    for rho in rho_:
        Javg__ = []
        grad__ = []
        for _ in range(n_repeat):
            u0 = u0_rand()
            Javg = only_for_J_forwardEuler(u0, parameter, nstep, 
                    runup_step)
            grad = np.nan
            print(rho, Javg, grad)
            Javg__.append(Javg)
            grad__.append(grad)
        Javg_.append(np.array(Javg__)) 
        grad_.append(np.array(grad__))
    Javg_ = np.array(Javg_)
    grad_ = np.array(grad_)
    pickle.dump((Javg_, grad_, rho_, sigma, M_modes, dt,\
            nstep, K_segment, runup_step, n_repeat), \
            open("onlyJ_change_rho.p", "wb"))    


def converge_T():
    # convergence of gradient to different trajectory length
    Javg_   = []
    grad_   = []
    K_segment_ = np.array([1e1, 2e1, 5e1, 1e2, 2e2, 5e2, 1e3, 2e3], dtype=int) 
    T_ = K_segment_ * dt * nstep_per_segment
    for K_segment, T in zip(K_segment_, T_):
        Javg__ = []
        grad__ = []
        for _ in range(n_repeat):
            u0 = u0_rand()
            Javg, grad, forward, interface, segment = nilsas_main(
                run_forward, run_adjoint, u0, M_modes,
                K_segment, nstep_per_segment, runup_step, 
                step_forward_euler, adjoint_step_explicit)
            print(T, Javg, grad)
            Javg__.append(Javg)
            grad__.append(grad)
        Javg_.append(np.array(Javg__)) 
        grad_.append(np.array(grad__))
    Javg_ = np.array(Javg_)
    grad_ = np.array(grad_)
    pickle.dump((Javg_, grad_, T_, K_segment_, rho, sigma, M_modes, dt, \
            nstep_per_segment, K_segment, runup_step, n_repeat), \
            open("changeK.p", "wb"))

        
def contour():
    # plot two parameters at the same time
    global rho, sigma
    rho_ = np.arange(29, 33.1, 1)
    sigma_ = np.arange(8, 11.1, 1)
    Javg_ = []
    grad_ = []
    for rho in rho_:
        for sigma in sigma_:
            with Pool(processes=4) as pool:
                results = pool.map(wrapped, range(n_repeat))
            Javg, grad = zip(*results)
            Javg_.append(Javg)
            grad_.append(grad)
    Javg_ = np.array(Javg_).reshape((len(rho_), len(sigma_), n_repeat))
    grad_ = np.array(grad_).reshape(Javg_.shape+(-1,))
    print(Javg_.shape)
    print(grad_.shape)
    pickle.dump(
            (Javg_, grad_, rho_, sigma_, M_modes, dt, nstep_per_segment, \
            K_segment, runup_step, n_repeat), \
            open("rho_sig.p", "wb"))
   

def contour_J():
    # only compute objective byt not gradient
    global rho, sigma
    Javg_   = []
    grad_   = []
    rho_ = np.arange(28.75, 33.3, 0.25)
    sigma_ = np.arange(7.75, 12.3, 0.25)
    rho_, sigma_ = np.meshgrid(rho_, sigma_)
    runup_step = 40000
    nstep = 100000
    n_repeat = 20
    for rho, sigma in zip(rho_.flatten(), sigma_.flatten()):
        Javg__ = []
        grad__ = []
        for _ in range(n_repeat):
            u0 = u0_rand()
            Javg = only_for_J_forwardEuler(u0, nstep, runup_step)
            grad = np.nan
            print(rho, sigma, Javg, grad)
            Javg__.append(Javg)
            grad__.append(grad)
        Javg_.append(np.array(Javg__)) 
        grad_.append(np.array(grad__))
    Javg_ = np.array(Javg_).reshape(rho_.shape+(n_repeat,))
    grad_ = np.array(grad_).reshape(rho_.shape+(n_repeat,-1))
    pickle.dump(
            (Javg_, grad_, rho_, sigma_, M_modes, dt, nstep, \
            K_segment, runup_step, n_repeat), \
            open("onlyJ_rho_sig.p", "wb"))    
    

if __name__ == '__main__': # pragma: no cover
    # all_info()
    contour()
    # change_rho()
    pass

