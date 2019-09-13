# The Forward class contains all information generated in the feedforward,
# such as J, f, Ju, u...
from pdb import set_trace
import numpy as np

class Forward:

    def __init__(self):
        # there is 1 objective and ns parameters
        # u:    shape(K, nstep_per_segment, m)
        # f:    shape(K, nstep_per_segment, m)
        # fu:   shape(K, nstep_per_segment, m, m)
        # fs:   shape(K, nstep_per_segment, ns, m)
        # J:    shape(K, nstep_per_segment,)
        # Ju:   shape(K, nstep_per_segment, m)
        # Js:   shape(K, nstep_per_segment, ns)
        self.u     = []
        self.f     = []
        self.fu    = []
        self.fs    = []
        self.J     = []
        self.Jtild = []
        self.Ju    = []
        self.Js    = []


    def run_allseg(self, u0, nstep_per_segment, K_segment, runup_steps, stepfunc, derivatives, dudt):
        # get u0 after runup time
        if runup_steps > 0:
            u, _, _, _, _, _, _ = self.run1seg(u0, runup_steps, stepfunc, derivatives, dudt)
        u0 = u[-1]

        u, f, fu, fs, J, Ju, Js = self.run1seg(u0, nstep_per_segment, stepfunc, derivatives, dudt)
        self.u.append(u)
        self.f.append(f)
        self.fu.append(fu)
        self.fs.append(fs)
        self.J.append(J)
        self.Ju.append(Ju)
        self.Js.append(Js)
        for i in range(1, K_segment):
            u, f, fu, fs, J, Ju, Js = self.run1seg(self.u[-1][-1], nstep_per_segment, stepfunc, derivatives, dudt, f0=self.f[-1][-1])
            self.u.append(u)
            self.f.append(f)
            self.fu.append(fu)
            self.fs.append(fs)
            self.J.append(J)
            self.Ju.append(Ju)
            self.Js.append(Js)

        self.u  = np.array(self.u)
        self.f  = np.array(self.f)
        self.fu = np.array(self.fu)
        self.fs = np.array(self.fs)
        self.J  = np.array(self.J)
        self.Ju = np.array(self.Ju)
        self.Js = np.array(self.Js)

        self.Jtild  = self.J - np.average(self.J)

        assert self.u.shape[:-1] == (K_segment, nstep_per_segment+1)
        assert self.f.shape[:-1] == (K_segment, nstep_per_segment+1)
        assert self.u.shape == self.f.shape
        assert np.allclose(self.u[1:,0], self.u[:-1,-1])
        assert np.allclose(self.f[1:,0], self.f[:-1,-1])


    def run1seg(self, u0, nstep, stepfunc, derivatives, dudt, f0=None):
        """
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
        assert len(u0) == m
        
        for i in range(1+nstep):
            if i == 0:
                u[i] = u0
                if f0 is None:
                    f[i] = dudt(u0)
                else:
                    f[i] = f0
            else:
                u[i], f[i] = stepfunc(u[i-1], f[i-1], fu[i-1])
            J[i], fu[i], Ju[i], fs[i] = derivatives(u[i])

        return u, f, fu, fs, J, Ju, Js
