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
        self.Ju    = []
        self.Js    = []

    def run(self, run_forward, u0, nstep_per_segment, K_segments):
        u, f, fu, fs, J, Ju, Js = run_forward(u0, nstep_per_segment)
        self.u.append(u)
        self.f.append(f)
        self.fu.append(fu)
        self.fs.append(fs)
        self.J.append(J)
        self.Ju.append(Ju)
        self.Js.append(Js)
        for i in range(1, K_segments):
            u, f, fu, fs, J, Ju, Js = run_forward(self.u[-1][-1], nstep_per_segment, f0 = self.f[-1][-1])
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
