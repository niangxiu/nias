from __future__ import division
import sys
import numpy as np
# import pascal_lite as pascal
# from multiprocessing import Pool
from pdb import set_trace

class TimeDilationBase:
    def contribution(self, v):
        if self.dxdt is None:
            if len(v.shape) == 0:
                return pascal.dot(v, v*0)
            else:
                return pascal.dot(v, v[0]*0)
        else:
            return pascal.dot(v, self.dxdt) / pascal.dot(self.dxdt, self.dxdt)

    def project(self, v):
        if self.dxdt_normalized is None:
            return v
        else:
            dv = pascal.outer(pascal.dot(v, self.dxdt_normalized), self.dxdt_normalized)
            return v - dv.reshape(v.shape)

class TimeDilationExact(TimeDilationBase):
    def __init__(self, run_ddt, u0, parameter):
        if run_ddt is not 0:
            self.dxdt = run_ddt(u0, parameter)
            self.dxdt_normalized = self.dxdt / linalg.norm(self.dxdt)
        else:
            self.dxdt = None
            self.dxdt_normalized = None

class TimeDilation(TimeDilationBase):
    order_of_accuracy = 3

    def __init__(self, run, u0, parameter, run_id,
                 simultaneous_runs, interprocess):
        threads = Pool(simultaneous_runs)
        res = []
        for steps in range(1, self.order_of_accuracy + 1):
            run_id_steps = run_id + '_{0}steps'.format(steps)
            # set_trace()
            res.append(threads.apply_async(
                run, (u0.field, parameter, steps, run_id_steps, interprocess)))
        
        u = [res_i.get()[0] for res_i in res]
        u = [u0] + [pascal.symbolic_array(field=ui) for ui in u]

        threads.close()
        threads.join()
        self.dxdt = compute_dxdt(u)
        self.dxdt_normalized = self.dxdt / pascal.norm(self.dxdt)
