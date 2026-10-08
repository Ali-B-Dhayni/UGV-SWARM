from dataclasses import dataclass
from typing import TYPE_CHECKING
import numpy as np
from c_jacobians import Ac, Bc, C, D
from fcont import observation, next_state
from d_jacobians import discrete_jacobians

if TYPE_CHECKING: 
    from ssm import SSM


@dataclass
class Linearisation:
    Ac  : np.ndarray
    Bc  : np.ndarray
    C   : np.ndarray
    D   : np.ndarray
    e   : np.ndarray
    A   : np.ndarray | None = None
    B   : np.ndarray | None = None
    c   : np.ndarray | None = None


def linearise(ssm: SSM, x, u, discrete=True) -> Linearisation:
    x, u = np.asarray(x, float), np.asarray(u, float)
    A_c, B_c, C_m, D_m = Ac(ssm, x, u), Bc(ssm, x, u), C(ssm, x, u), D(ssm, x, u)

    lin = Linearisation(A_c, B_c, C_m, D_m, e=observation(ssm, x, u) - C_m @ x - D_m @ u)
    if discrete:
        lin.A, lin.B = discrete_jacobians(ssm, x, u)
        lin.c = next_state(ssm, x, u) - lin.A @ x - lin.B @ u
    return lin
