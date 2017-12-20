import numpy as np

def get_C_cts(w):
    # compute the covariant matrix using w on all time steps
    # w is on a segment, shape (nstep_per_segment, M, m)
    assert w.ndim == 3
    assert w.shape[1] <= w.shape[2]
    M_modes = w.shape[1]
    C = np.zeros([M_modes, M_modes])
    for i in range M_modes:
        for j in range M_modes:
            C[i,j] = np.sum( w[:, i, :] * w[:, j, :] )
            C[j,i] = C[i,j]
    return C


def get_d_cts(w, p)
    # p is either ystar or vstar, shape (nstep_per_segment, m)
    assert p.ndim == 2
    assert w.shape[2] == p.shape[1]
    M_modes = w.shape[1]
    d = np.zeros(M_modes)
    for i in range(M_modes):
        d[i] = np.sum( w[:, i, :] * p )
    return d


class Segment:

    def __init__(self):
        # w:        shape(K, nstep_per_segment, M, m)  
        # yst, vst: shape(K, nstep_per_segment, m)
        # C:        shape(K, nstep_per_segment, M, M)
        # dy, dv:   shape(K, nstep_per_segment, M)
        
        self.w      = np.array([])
        self.yst    = np.array([])
        self.vst    = np.array([])
        self.C      = np.array([])
        self.dy     = np.array([])
        self.dv     = np.array([])


    def run1seg(self, run_adjoint, interface, forward):
        w_tmn   = interface.Q[0]
        yst_tmn = interface.yst_left[0]
        vst_tmn = interface.vst_left[0]
        fu      = forward.fu[0]
        Ju      = forward.Ju[0]

        w, yst, vst = run_adjoint(w_tmn, yst_tmn, vst_tmn, fu, Ju)
        C   = get_C_cts(w)
        dy  = get_d_cts(w, yst)
        dv  = get_d_cts(w, vst)

        self.w      = np.concatenate((w, self.w),     axis=0)
        self.yst    = np.concatenate((yst, self.yst), axis=0)
        self.vst    = np.concatenate((vst, self.vst), axis=0)
        self.C      = np.concatenate((C, self.C),     axis=0)
        self.dy     = np.concatenate((dy, self.dy),   axis=0)
        self.dv     = np.concatenate((dv, self.dv),   axis=0)
