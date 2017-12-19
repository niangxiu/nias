import numpy as np
from scipy import sparse
import scipy.sparse.linalg as splinalg

from .timeseries import windowed_mean
from .utility import qr_transpose

class AdjointShadow:
    def __init__(self):
        self.Rs     = []
        self.bvs    = []
        self.bys    = []
        self.Cs     = []

    def K_egments(self):
        assert len(self.Rs) == len(self.bs)
        return len(self.Rs)

    def m_modes(self):
        return self.Rs[0].shape[0]

    def rescale(self, W, vih):
        Q, R = qr_transpose(W)
        b = np.dot(Q, vih)
        W = Q
        vih = vih - np.dot(b, Q)

        self.Rs.insert(0, R)
        self.bs.insert(0, b)
        return V, v

    def solve_nilsas(self):
        # todo:
        # this function solves the NILSAS problem, which is a least squares problem
        R, b = np.array(self.Rs), np.array(self.bs)
        assert R.ndim == 3 and b.ndim == 2
        assert R.shape[0] == b.shape[0]
        assert R.shape[1] == R.shape[2] == b.shape[1]
        nseg, subdim = b.shape
        eyes = np.eye(subdim, subdim) * np.ones([nseg, 1, 1])
        matrix_shape = (subdim * nseg, subdim * (nseg+1))
        I = sparse.bsr_matrix((eyes, np.r_[1:nseg+1], np.r_[:nseg+1]))
        D = sparse.bsr_matrix((R, np.r_[:nseg], np.r_[:nseg+1]), shape=matrix_shape)
        B = (D - I).tocsr()
        Schur = B * B.T #+ 1E-5 * sparse.eye(B.shape[0])
        alpha = -(B.T * splinalg.spsolve(Schur, np.ravel(b)))
        return alpha.reshape([nseg+1,-1])[:-1]

    def lyapunov_exponents(self, segment_range=None):
        R = np.array(self.Rs)
        if segment_range is not None:
            R = R[slice(*segment_range)]
        i = np.arange(self.m_modes())
        diags = R[:,i,i]
        return np.log(abs(diags))

    def lyapunov_covariant_vectors(self):
        # might need further check
        exponents = self.lyapunov_exponents().mean(0)
        multiplier = np.exp(exponents)
        Ci = np.eye(self.m_modes())
        C = [Ci]
        for Ri in self.Rs:
            Ci = np.linalg.solve(Ri, Ci) * multiplier
            C.insert(0, Ci)
        C = np.array(C)
        return C # we do not roll axis here, unlike in Qiqi's fds!

    def lyapunov_covariant_magnitude_and_sin_angle(self):
        # might need further check
        v = self.lyapunov_covariant_vectors()
        v_magnitude = np.sqrt((v**2).sum(2))
        vv = (v[:,np.newaxis] * v[np.newaxis,:]).sum(3)
        cos_angle = (vv / v_magnitude).transpose([1,0,2]) / v_magnitude
        i = np.arange(cos_angle.shape[0])
        cos_angle[i,i,:] = 1
        sin_angle = np.sqrt(1 - cos_angle**2)
        return v_magnitude, sin_angle


