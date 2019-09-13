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


def dudt(u):
    [x, y, z] = u
    return np.array([sigma*(y-x), x*(rho-z)-y, x*y-beta*z])


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


def step_forward_euler(u, f, fu):
    u_next = u + f * dt
    return u_next, dudt(u_next)


def step_RK4(u, f, fu):
    k0 = dt * dudt(u, rho, sigma) 
    k1 = dt * dudt(u + 0.5 * k0, rho, sigma)
    k2 = dt * dudt(u + 0.5 * k1, rho, sigma)
    k3 = dt * dudt(u + k2, rho, sigma)
    u_next = u + (k0 + 2*k1 + 2*k2 + k3) / 6.0
    return u_next, dudt(u_next)


def adjoint_step_explicit(fu, Ju, adjall):
    adjall_next = (np.dot(fu.T, adjall.T) * dt + adjall.T).T
    adjall_next[-1] += Ju * dt
    return adjall_next


def adjoint_step_implicit(fu, Ju, adjall):
    adjall[-1] += Ju * dt
    adjall_next = (np.linalg.solve(np.eye(3)-fu.T*dt, adjall.T)).T
    return adjall_next


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


def wrapped_nilsasmain(keep_array=False): 
    u0s = [u0_rand() for i in range(n_repeat)]
    arguments = [(u0, M_modes, K_segment, nstep_per_segment, runup_step,
            step_forward_euler, adjoint_step_explicit, derivatives, dudt) for u0 in u0s]
    with Pool(processes=n_repeat) as pool:
        results = pool.starmap(nilsas_main, arguments)
    Javg_, grad_, forward_, interface_, segment_ = zip(*results)
    [print(rho, sigma, Javg, grad) for Javg,grad in zip(Javg_,grad_)]
    if keep_array:
        return np.array(Javg_), np.array(grad_)
    else: 
        return np.mean(Javg_), np.mean(grad_, axis=0)


def wrapped_onlyJ(keep_array=False):
    u0s = [u0_rand() for i in range(n_repeat)]
    arguments = [(u0, nstep_only_J, runup_step) for u0 in u0s]
    with Pool(processes=n_repeat) as pool:
        results = pool.starmap(only_for_J_forwardEuler, arguments)
    print(rho, sigma, results)
    if keep_array:
        return np.array(results)
    else:
        return np.mean(results)


def converge_T():
    # convergence of gradient to different trajectory length
    global K_segment
    Javg_   = []
    grad_   = []
    K_segment_ = np.array([1e1, 2e1, 5e1, 1e2, 2e2, 5e2], dtype=int) 
    T_ = K_segment_ * dt * nstep_per_segment
    for K_segment in K_segment_:
        Javg, grad = wrapped_nilsasmain(keep_array=True)
        Javg_.append(Javg) 
        grad_.append(grad)
        print(K_segment, Javg, grad)
    pickle.dump((Javg_, grad_, T_, K_segment_, rho, sigma, M_modes, dt, \
            nstep_per_segment, K_segment, runup_step, n_repeat), \
            open("change_T.p", "wb"))

        
def contour():
    # plot two parameters at the same time
    global rho, sigma
    rho_ = np.arange(29, 33.1, 1)
    sigma_ = np.arange(8, 12.1, 1)
    Javg_ = []
    grad_ = []
    for rho in rho_:
        for sigma in sigma_:
            Javg, grad = wrapped_nilsasmain()
            Javg_.append(Javg)
            grad_.append(grad)
    Javg_ = np.array(Javg_).reshape((len(rho_), len(sigma_)))
    grad_ = np.array(grad_).reshape(Javg_.shape+(-1,))
    print(Javg_.shape)
    print(grad_.shape)
    pickle.dump((Javg_, grad_, rho_, sigma_, M_modes, dt, nstep_per_segment, \
            K_segment, runup_step, n_repeat), open("rho_sig.p", "wb"))
   

def contour_J():
    # only compute objective byt not gradient
    global rho, sigma
    Javg_   = []
    rho_ = np.arange(28.5, 33.51, 0.5)
    sigma_ = np.arange(7.5, 12.51, 0.5)
    for rho in rho_:
        for sigma in sigma_:
            Javg_.append(wrapped_onlyJ()) 
    Javg_ = np.array(Javg_).reshape((len(rho_), len(sigma_)))
    pickle.dump((Javg_, rho_, sigma_, M_modes, dt, nstep_only_J, \
            K_segment, runup_step, n_repeat), open("onlyJ_rho_sig.p", "wb"))    

    
def change_rho():
    # grad for different rho
    global rho
    Javg_   = []
    grad_   = []
    rho_ = np.arange(25, 50.1, 1)
    for rho in rho_:
        Javg, grad = wrapped_nilsasmain(keep_array=True)
        Javg_.append(Javg) 
        grad_.append(grad)
    pickle.dump((Javg_, grad_, rho_, sigma, M_modes, dt, \
            nstep_per_segment, K_segment, runup_step, n_repeat), \
            open("change_rho.p", "wb"))


def change_rho_J():
    global rho
    Javg_   = []
    rho_ = np.arange(24.75, 50.31, 0.5) 
    for rho in rho_:
        Javg_.append(wrapped_onlyJ(keep_array=True)) 
    pickle.dump((Javg_, rho_, sigma, M_modes, dt,\
            nstep_only_J, K_segment, runup_step, n_repeat), \
            open("onlyJ_change_rho.p", "wb"))    


def all_info():
    # generate all info
    u0 = u0_rand()
    Javg, grad, forward, interface, segment = nilsas_main(
        u0, M_modes, K_segment, nstep_per_segment, runup_step, 
        step_forward_euler, adjoint_step_explicit, derivatives, dudt)
    pickle.dump((Javg, grad, forward, interface, segment, dt, K_segment,
            nstep_per_segment), open("all_info_segment.p", "wb"))



beta = 8/3.0
n_repeat = 4
dt = 0.001 # largest step allowed for PTA forward Euler: dt = 0.001
nstep_per_segment = 200
M_modes = 2
K_segment = 200
runup_step = 10000
rho = 28
sigma = 10
nstep_only_J = 10000


if __name__ == '__main__': # pragma: no cover
    # converge_T()
    # contour()
    # contour_J()
    # change_rho()
    # change_rho_J()
    all_info()
    pass
