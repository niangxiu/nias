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


def get_d_cts_wp(w, p):
    # p is either vstar or f, shape (nstep_per_segment, m)
    assert p.ndim == 2
    assert w.shape[2] == p.shape[1]
    M_modes = w.shape[1]
    d = np.zeros(M_modes)
    for i in range(M_modes):
        d[i] = np.sum( w[:, i, :] * p )
    return d


def get_d_cts_vf(v, f):
    # vstar and f, shape (nstep_per_segment, m)
    assert v.shape == f.shape
    return np.sum(v*f)


class Segment:

    def __init__(self):
        # w: shape(K, nstep_per_segment, M, m)  
        # vst, v: shape(K, nstep_per_segment, m)
        # C: shape(K, M, M)
        # dwv, dwf: shape(K, M)
        # dvf: shape(K)
        
        self.w = np.array([])
        self.vst = np.array([])
        self.C = np.array([])
        self.dwv = np.array([]) 
        self.dwf = np.array([]) 
        self.dvf = np.array([]) 


    def run1seg(self, run_adjoint, interface, forward, dt, stepfunc):
        j_current_segment = -(self.w.shape[0] + 1)

        w_tmn = interface.Q[0]
        vst_tmn = interface.vst_left[0]
        fu = forward.fu[j_current_segment]
        Ju = forward.Ju[j_current_segment]
        f = forward.f[j_current_segment]

        w, vst = run_adjoint(w_tmn, vst_tmn, fu, Ju, dt, stepfunc)
        C = get_C_cts(w)
        dwv = get_d_cts_wp(w, vst)
        dwf = get_d_cts_wp(w, f)
        dvf = get_d_cts_vf(vst, f)
   
        self.w = stackv(w, self.w)
        self.vst = stackv(vst, self.vst)
        self.C = stackv(C, self.C)
        self.dwv = stackv(dwv, self.dwv)
        self.dwf = stackv(dwf, self.dwf)
        self.dvf = stackv(dvf, self.dvf)

    
    def get_v(self, av, f, Jtild):
        assert av.shape == self.dwv.shape
        self.v = self.vst\
                + (self.w * av[:,np.newaxis,:,np.newaxis]).sum(axis=-2)
