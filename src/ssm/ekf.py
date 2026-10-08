from typing import TYPE_CHECKING
import numpy as np
from params import PARAMS as p
from linearise import linearise
from utils import wrap
from fcont import next_state, observation
from d_jacobians import discrete_jacobians

if TYPE_CHECKING:
    from ssm import SSM


def ekf(ssm: SSM, U, Y, x0, P0):
    T = len(Y)
    Xs, Ps = np.zeros((T, p.NX)), np.zeros((T, p.NX, p.NX))

    x, P, nll = np.array(x0, float), np.array(P0, float), 0.0
    Qd = ssm.Q * ssm.dt
    for k in range(T):
        seen = ~np.isnan(Y[k])
        if seen.any():
            H = linearise(ssm, x, U[k], discrete=False).C[seen]
            innov = Y[k] - observation(ssm, x, U[k])
            if seen[p.O.GPS_COURSE]:
                innov[p.O.GPS_COURSE] = wrap(innov[p.O.GPS_COURSE])
            innov = innov[seen]
            Rm = ssm.R[np.ix_(seen, seen)]
            S_ = H @ P @ H.T + Rm
            K = np.linalg.solve(S_, H @ P).T
            x = x + K @ innov
            I_KH = np.eye(p.NX) - K @ H
            P = I_KH @ P @ I_KH.T + K @ Rm @ K.T
            nll += 0.5 * (innov @ np.linalg.solve(S_, innov) + np.linalg.slogdet(S_)[1])
        Xs[k], Ps[k] = x, P

        if k < T - 1:
            A = discrete_jacobians(ssm, x, U[k])[0]
            x = next_state(ssm, x, U[k])
            P = A @ P @ A.T + Qd

    return Xs, Ps, nll
