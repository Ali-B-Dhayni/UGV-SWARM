from ugv import UGV
from utils import  wrap
import numpy as np
from params import PARAMS as p
from fcont import next_state, observation
from ekf import ekf

class SSM:
    def __init__(self, ugv:UGV | None =None, dt:float=0.1, process_noise:dict | None =None, sensor_bias:dict | None =None):
        self.ugv = ugv
        if ugv is None:
            self.ugv = UGV()
        self.dt = dt
        q = {**p.PROCESS_NOISE, **(process_noise or {})}
        s = {**p.SENSOR_BIAS, **(sensor_bias or {})}
        self.Q = np.diag([q[n] for n in p.STATE_VARS])
        self.R = np.diag([s[n] ** 2 for n in p.OBSERVATTIONS])

    def initial_state(self, psi0=0.0, v0=0.0):
        x0 = np.zeros(p.NX)
        x0[p.S.PSI] = psi0
        x0[p.S.VX] =  v0
        x0[p.S.KW] = 1.0
        return x0

    def simulate(self, x0, U, rng=None, noise=True, delta_profile=None):
        rng = rng or np.random.default_rng(0)
        T = len(U)
        X, Y = np.zeros((T, p.NX)), np.zeros((T, p.NY))
        w_std = np.sqrt(np.diag(self.Q) * self.dt)
        v_std = np.sqrt(np.diag(self.R))
        x = np.array(x0, float)
        for k in range(T):
            if delta_profile is not None:
                x[p.S.DELTA] = delta_profile[k]
            X[k] = x
            Y[k] = observation(self, x, U[k]) + (rng.normal(size=p.NY) * v_std if noise else 0.0)
            x = next_state(self, x, U[k]) + (rng.normal(size=p.NX) * w_std if noise else 0.0)
        return X, Y



if __name__ == "__main__":
    model = SSM()

    T = 1200
    t = np.arange(T) * model.dt
    U = np.column_stack([np.where(t < 15, 35.0, 12.0 + 4 * np.sin(2 * np.pi * t / 40)),
                         np.where((t > 80) & (t < 85), 15.0, 0.0),                    
                         np.where(t < 4, 9.0, np.where(t < 9, 6.0, 4.0))])           
    steering = 0.04 * np.sin(2 * np.pi * t / 25) * (t > 10)                         

    x0_true = model.initial_state(psi0=0.5, v0=2.0)
    X, Y = model.simulate(x0_true, U, rng=np.random.default_rng(1), delta_profile=steering)
    Y[300:450, [p.O.GPS_X, p.O.GPS_Y, p.O.GPS_SPEED, p.O.GPS_COURSE]] = np.nan            

    x0 = x0_true.copy()
    x0[[p.S.PX, p.S.PY, p.S.PSI, p.S.VX, p.S.KW]] += [3, -3, 0.2, 0.5, 0.02] 
    P0 = np.diag([10, 10, 0.2, 1, 0.5, 0.1, 0.05, 0.05, 0.02, 0.05, 0.1]) ** 2
    Xf, Pf, nll = ekf(model, U, Y, x0, P0)
    rmse = np.sqrt(np.mean((Xf - X) ** 2, axis=0))
    print("EKF negative log-likelihood:", round(nll, 1))
    for name, e in zip(p.STATE_VARS, rmse):
        print(f"  RMSE {name:5s}: {e:.4f}")
