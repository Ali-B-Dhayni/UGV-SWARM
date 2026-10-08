from typing import TYPE_CHECKING
import numpy as np
from params import PARAMS as p
from fcont import state_derivative
from c_jacobians import Ac, Bc

if TYPE_CHECKING:
    from ssm import SSM


def discrete_jacobians(ssm: SSM, x, u):
    h = ssm.dt / ssm.ugv.nsub
    Id = np.eye(p.NX)
    A = Id.copy()
    B = np.zeros((p.NX, p.NU))
    x = np.asarray(x, float)
    for _ in range(ssm.ugv.nsub):
        k1 = state_derivative(ssm, x, u)
        x2 = x + 0.5 * h * k1
        k2 = state_derivative(ssm, x2, u)
        x3 = x + 0.5 * h * k2
        k3 = state_derivative(ssm, x3, u)
        x4 = x + h * k3
        k4 = state_derivative(ssm, x4, u)

        A1, A2, A3, A4 = (Ac(ssm, xi, u) for xi in (x, x2, x3, x4))
        B_c = Bc(ssm, x, u) 

        K1 = A1
        K2 = A2 @ (Id + 0.5 * h * K1)
        K3 = A3 @ (Id + 0.5 * h * K2)
        K4 = A4 @ (Id + h * K3)
        L1 = B_c
        L2 = B_c + A2 @ (0.5 * h * L1)
        L3 = B_c + A3 @ (0.5 * h * L2)
        L4 = B_c + A4 @ (h * L3)
        step_A = Id + h / 6 * (K1 + 2 * K2 + 2 * K3 + K4)
        step_B = h / 6 * (L1 + 2 * L2 + 2 * L3 + L4)
        A, B = step_A @ A, step_A @ B + step_B
        x = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return A, B
