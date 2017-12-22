import numpy as np
from .utility import stackv

def get_C_cts(w):
    # compute the covariant matrix using w on all time steps
    # w is on a segment, shape (nstep_per_segment, M, m)
    assert w.ndim == 3
    assert w.shape[1] <= w.shape[2]
    M_modes = w.shape[1]
    C = np.zeros([M_modes, M_modes])
    for i in range(M_modes):
        for j in range(M_modes):
            C[i,j] = np.sum( w[:, i, :] * w[:, j, :] )
            C[j,i] = C[i,j]
    return C


def get_d_cts(w, p):
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
        # C:        shape(K, M, M)
        # dy, dv:   shape(K, M)
        
        self.w      = np.array([])
        self.yst    = np.array([])
        self.vst    = np.array([])
        self.C      = np.array([])
        self.dy     = np.array([])
        self.dv     = np.array([])


    def run1seg(self, run_adjoint, interface, forward, dt):
        j_current_segment = -(self.w.shape[0] + 1)

        w_tmn   = interface.Q[0]
        yst_tmn = interface.yst_left[0]
        vst_tmn = interface.vst_left[0]
        fu      = forward.fu[j_current_segment]
        Ju      = forward.Ju[j_current_segment]

        w, yst, vst = run_adjoint(w_tmn, yst_tmn, vst_tmn, fu, Ju, dt)
        C   = get_C_cts(w)
        dy  = get_d_cts(w, yst)
        dv  = get_d_cts(w, vst)
   
        self.w      = stackv(w,     self.w)
        self.yst    = stackv(yst,   self.yst)
        self.vst    = stackv(vst,   self.vst)
        self.C      = stackv(C,     self.C)
        self.dy     = stackv(dy,    self.dy)
        self.dv     = stackv(dv,    self.dv)
