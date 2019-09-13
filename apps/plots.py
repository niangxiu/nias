import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle
import sys
from pdb import set_trace
sys.path.append("../")
from nilsas.nilsas import nilsas_main


plt.rc('axes', labelsize='xx-large',  labelpad=12)
plt.rc('xtick', labelsize='xx-large')
plt.rc('ytick', labelsize='xx-large')
plt.rc('legend', fontsize='xx-large')
plt.rc('font', family='sans-serif')


def T_J_dJ():
    # plot T vs. J, dJdrho, dJdsigma
    Javg_, grad_, T_, K_segment_, rho, sigma, M_modes, dt, \
            nstep_per_segment, K_segment, runup_step, n_repeat \
            = pickle.load( open( "change_T.p", "rb" ) )
    x = np.array([T_[0], T_[-1]])
    Javg_ = np.array(Javg_)
    grad_ = np.array(grad_)

    print("plot T ~ J")
    print(Javg_.shape)
    print(Javg_[0])
    plt.semilogx(T_, Javg_, 'k.')
    plt.xlabel('$T$')
    plt.ylabel('$J_{avg}$')
    plt.tight_layout()
    plt.savefig('T_J.png')
    plt.close()

    plt.loglog(T_, np.std(Javg_, axis=1), 'k.')
    plt.loglog(x, x**-0.5, 'k--')
    plt.xlabel('$T$')
    plt.ylabel('std $\left( \, J_{avg} \, \\right)$')
    plt.tight_layout()
    plt.savefig('T_J_std.png')
    plt.close()

    print("plot T ~ dJ/drho")
    plt.semilogx(T_, grad_[:,:,0], 'k.')
    plt.xlabel('$T$')
    plt.ylabel('$\\partial J_{avg} \\,/\\, \\partial \\rho$')
    plt.tight_layout()
    plt.savefig('T_grad_rho.png')
    plt.close()

    plt.loglog(T_, np.std(grad_[:,:,0], axis=1), 'k.')
    plt.loglog(x, x**-0.5, 'k--')
    plt.xlabel('$T$')
    plt.ylabel('std $\left( \, \\partial J_{avg} \\,/\\, \\partial \\rho \, \\right)$')
    plt.tight_layout()
    plt.savefig('T_grad_rho_std.png')
    plt.close()

    print("plot T ~ dJ/dsigma")
    plt.semilogx(T_, grad_[:,:,1], 'k.')
    plt.xlabel('$T$')
    plt.ylabel('$\\partial J_{avg} \\,/\\, \\partial \sigma$')
    plt.tight_layout()
    plt.savefig('T_grad_sigma.png')
    plt.close()

    plt.loglog(T_, np.std(grad_[:,:,1], axis=1), 'k.')
    plt.loglog(x, x**-0.5, 'k--')
    plt.xlabel('$T$')
    plt.ylabel('std $\left( \, \\partial J_{avg} \\,/\\, \\partial \sigma \, \\right)$')
    plt.tight_layout()
    plt.savefig('T_grad_sigma_std.png')
    plt.close()


def contours():
    # contourf, rho and sigma ~ J, gradient direction
    print('plot contour: rho,sig ~ J')
    Javg_, rho_, sigma_, M_modes, dt, nstep, \
            K_segment, runup_step, n_repeat \
            = pickle.load( open("onlyJ_rho_sig.p", "rb"))
    sigma_, rho_ = np.meshgrid(sigma_, rho_)

    fig = plt.figure(figsize=(10,8))
    ax = plt.axes()
    CS = plt.contourf(rho_, sigma_, Javg_, 12,
              cmap=plt.cm.bone, origin='lower')
    cbar = plt.colorbar(CS)
    cbar.ax.set_ylabel('$J_{avg}$')
    plt.xlabel('$\\rho$')
    plt.ylabel('$\sigma$')

    Javg_, grad_, rho_, sigma_, M_modes, dt, nstep_per_segment, \
            K_segment, runup_step, n_repeat \
            = pickle.load( open("rho_sig.p", "rb") )
    sigma_, rho_ = np.meshgrid(sigma_, rho_)
    print(grad_.shape)

    plt.scatter(rho_, sigma_, color='k', s=30)
    Q = plt.quiver(rho_, sigma_,  grad_[...,0],  grad_[...,1], units='x',
            pivot='tail', width=0.02, color='r', scale=5)

    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plt.savefig('contour_rho_sig_J.png')
    plt.close()


def rho_J():
    # change rho plots
    print('plot rho ~ J')
    Javg_, rho_, sigma, M_modes, dt,\
                nstep, K_segment, runup_step, n_repeat \
                = pickle.load( open( "onlyJ_change_rho.p", "rb" ) )
    plt.plot(rho_, Javg_, 'k.')
    plt.xlabel('$\\rho$')
    plt.ylabel('$J_{avg}$')
    plt.tight_layout()
    plt.savefig('rho_J.png')
    plt.close()


def rho_dJdrho():
    Javg_, grad_, rho_, sigma, M_modes, dt,\
                nstep, K_segment, runup_step, n_repeat \
                = pickle.load( open( "change_rho.p", "rb" ) )
    Javg_ = np.array(Javg_)
    grad_ = np.array(grad_)
    print("plot rho ~ dJdrho")
    print(Javg_.shape)
    print(nstep, K_segment, dt, nstep*K_segment*dt)
    plt.plot(rho_, grad_[...,0], 'k.')
    plt.xlabel('$\\rho$')
    plt.ylabel('$dJ_{avg}/d\\rho$')
    plt.tight_layout()
    plt.savefig('rho_dJdrho.png')
    plt.close()


def asd():
    # plot adjoint shadowing direction, first a long trajec, then a zoom-in
    Javg, grad, forward, interface, segment, dt, K_segment, nstep_per_segment \
            = pickle.load(open("all_info_segment.p", "rb"))
    time_ref = dt * np.arange(nstep_per_segment+1)
    DeltaT = dt * nstep_per_segment
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,5), sharey=True)

    for i in range (K_segment):
        time = i*DeltaT + time_ref
        vnorm = np.linalg.norm(segment.v[i], axis=-1)
        ax1.plot(time, vnorm, 'k')
    ax1.set(xlabel='$time$')
    ax1.set(ylabel='$\\| \overline{v} \\|$')
    plt.tight_layout()

    zoom_range = range(95,105)
    for i in zoom_range:
        time = i*DeltaT + time_ref
        vnorm = np.linalg.norm(segment.v[i], axis=-1)
        ax2.plot(time, vnorm, 'k')
        if i != zoom_range[-1]:
            ax2.plot([time[-1], time[-1]], [vnorm[-1]-0.2, vnorm[-1]+0.2], 'k--')
    ax2.set(xlabel='$time$')
    plt.tight_layout()

    plt.savefig('adjoint_shadowing_direction.png')
    plt.close()


if __name__ == '__main__':
    # T_J_dJ()
    # contours()
    # rho_J()
    # rho_dJdrho()
    asd()
    pass
