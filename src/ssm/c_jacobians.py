from typing import TYPE_CHECKING
import numpy as np
from params import PARAMS as p

if TYPE_CHECKING:
    from ssm import SSM


def _extract_states(ssm: SSM, x, u):
    ugv = ssm.ugv
    psi, vx, vy, r, k_w, delta = x[p.S.PSI], x[p.S.VX], x[p.S.VY], x[p.S.R], x[p.S.KW], x[p.S.DELTA]
    return psi, vx, vy, r, k_w, delta, ugv


def Ac(ssm: SSM, x, u):
    psi, vx, vy, r, k_w, delta, ugv = _extract_states(ssm, x, u)
    k1, k2, j2, j3 = ugv.BM_lateral_coefficients()
    vs = np.sqrt(vx ** 2 + ugv.vmin ** 2)
    vs3 = vs ** 3
    dax_dvx = -2 * ugv.cdr * abs(vx) - ugv.cro * (1 - np.tanh(vx / 0.5) ** 2) / 0.5
    A = np.zeros((p.NX, p.NX))
    A[p.S.PX, p.S.PSI] = -vx * np.sin(psi) - vy * np.cos(psi)
    A[p.S.PX, p.S.VX], A[p.S.PX, p.S.VY] = np.cos(psi), -np.sin(psi)
    A[p.S.PY, p.S.PSI] = vx * np.cos(psi) - vy * np.sin(psi)
    A[p.S.PY, p.S.VX], A[p.S.PY, p.S.VY] = np.sin(psi), np.cos(psi)
    A[p.S.PSI, p.S.R] = 1.0
    A[p.S.VX, p.S.VX], A[p.S.VX, p.S.VY], A[p.S.VX, p.S.R] = dax_dvx, r, vy
    A[p.S.VY, p.S.VX] = k1 * vy * vx / vs3 - r * (1 - k2 * vx / vs3)
    A[p.S.VY, p.S.VY] = -k1 / vs
    A[p.S.VY, p.S.R] = -(vx + k2 / vs)
    A[p.S.VY, p.S.DELTA] = ugv.Cf / ugv.m
    A[p.S.R, p.S.VX] = (j2 * vy * vx + j3 * r * vx) / vs3
    A[p.S.R, p.S.VY] = -j2 / vs
    A[p.S.R, p.S.R] = -j3 / vs
    A[p.S.R, p.S.DELTA] = ugv.Lf * ugv.Cf / ugv.Iz

    return A


def Bc(ssm: SSM, x, u):
    psi, vx, vy, r, k_w, delta, ugv = _extract_states(ssm, x, u)
    B = np.zeros((p.NX, p.NU))
    B[p.S.VX, p.I.PEDAL] = u[p.I.RATIO] / ugv.r_ref * ugv.kped
    B[p.S.VX, p.I.BRAKE] = -ugv.kbrk
    B[p.S.VX, p.I.RATIO] = ugv.kped * u[p.I.PEDAL] / ugv.r_ref

    return B


def C(ssm: SSM, x, u):
    psi, vx, vy, r, k_w, delta, ugv = _extract_states(ssm, x, u)

    Cm = np.zeros((p.NY, p.NX))
    Cm[p.O.GPS_X, p.S.PX] = Cm[p.O.GPS_Y, p.S.PY] = 1.0

    speed = np.hypot(vx, vy)
    if speed > 1e-6:
        Cm[p.O.GPS_SPEED, p.S.VX], Cm[p.O.GPS_SPEED, p.S.VY] = vx / speed, vy / speed

    vxm = max(vx, p.V_EPS)
    Cm[p.O.GPS_COURSE, p.S.PSI] = 1.0
    Cm[p.O.GPS_COURSE, p.S.VY] = vxm / (vxm ** 2 + vy ** 2)

    if vx > p.V_EPS:
        Cm[p.O.GPS_COURSE, p.S.VX] = -vy / (vxm ** 2 + vy ** 2)
    Cm[p.O.IND_SPEED, p.S.VX] = 1.0

    wr = ugv.R0 * k_w
    v_left, v_right = vx - r * ugv.tw / 2, vx + r * ugv.tw / 2
    cd, sd = np.cos(delta), np.sin(delta)
    side = vy + r * ugv.Lf

    A_c = Ac(ssm, x, u) 
    for row, v_hub, v_side, dv_dr in [(p.O.W_FL, v_left * cd + side * sd, v_left, -ugv.tw / 2),
                                      (p.O.W_FR, v_right * cd + side * sd, v_right, ugv.tw / 2)]:
        Cm[row, p.S.VX], Cm[row, p.S.VY] = cd / wr, sd / wr
        Cm[row, p.S.R] = (dv_dr * cd + ugv.Lf * sd) / wr
        Cm[row, p.S.KW] = -v_hub / wr / k_w
        Cm[row, p.S.DELTA] = (-v_side * sd + side * cd) / wr

    Cm[p.O.W_RL, p.S.VX], Cm[p.O.W_RL, p.S.R], Cm[p.O.W_RL, p.S.KW] = 1 / wr, -ugv.tw / 2 / wr, -v_left / wr / k_w
    Cm[p.O.W_RR, p.S.VX], Cm[p.O.W_RR, p.S.R], Cm[p.O.W_RR, p.S.KW] = 1 / wr, ugv.tw / 2 / wr, -v_right / wr / k_w
    Cm[p.O.YAW_RATE, p.S.R] = Cm[p.O.YAW_RATE, p.S.BR] = 1.0
    Cm[p.O.AX, p.S.VX], Cm[p.O.AX, p.S.BAX] = A_c[p.S.VX, p.S.VX], 1.0
    Cm[p.O.AY, p.S.VX] = A_c[p.S.VY, p.S.VX] + r
    Cm[p.O.AY, p.S.VY] = A_c[p.S.VY, p.S.VY]
    Cm[p.O.AY, p.S.R] = A_c[p.S.VY, p.S.R] + vx
    Cm[p.O.AY, p.S.DELTA] = A_c[p.S.VY, p.S.DELTA]
    Cm[p.O.AY, p.S.BAY] = 1.0
    Cm[p.O.STEER, p.S.DELTA] = ugv.sr

    return Cm


def D(ssm: SSM, x, u):
    Dm = np.zeros((p.NY, p.NU))
    B_c = Bc(ssm, x, u)
    Dm[p.O.AX, :] = B_c[p.S.VX, :]

    return Dm
