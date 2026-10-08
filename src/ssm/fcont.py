from typing import TYPE_CHECKING
import numpy as np
from params import PARAMS as p

if TYPE_CHECKING:    
    from ssm import SSM


def longitudinal_acceleration(ssm: SSM, x, u):
    ugv = ssm.ugv
    vx = x[p.S.VX]
    return ((u[p.I.RATIO] / ugv.r_ref) * ugv.kped * u[p.I.PEDAL] - ugv.kbrk * u[p.I.BRAKE]
            - ugv.cdr * vx * abs(vx) - ugv.cro * np.tanh(vx / 0.5))


def state_derivative(ssm: SSM, x, u):
    ugv = ssm.ugv
    psi, vx, vy, r, delta = x[p.S.PSI], x[p.S.VX], x[p.S.VY], x[p.S.R], x[p.S.DELTA]
    k1, k2, j2, j3 = ugv.BM_lateral_coefficients()
    vs = np.sqrt(vx ** 2 + ugv.vmin ** 2)
    xdot = np.zeros(p.NX)
    xdot[p.S.PX] = vx * np.cos(psi) - vy * np.sin(psi)
    xdot[p.S.PY] = vx * np.sin(psi) + vy * np.cos(psi)
    xdot[p.S.PSI] = r
    xdot[p.S.VX] = longitudinal_acceleration(ssm, x, u) + r * vy
    xdot[p.S.VY] = -k1 / vs * vy - (vx + k2 / vs) * r + ugv.Cf / ugv.m * delta
    xdot[p.S.R] = -j2 / vs * vy - j3 / vs * r + ugv.Lf * ugv.Cf / ugv.Iz * delta

    return xdot


def next_state(ssm: SSM, x, u):
    h = ssm.dt / ssm.ugv.nsub
    x = np.asarray(x, float)
    for _ in range(ssm.ugv.nsub):
        k1 = state_derivative(ssm, x, u)
        k2 = state_derivative(ssm, x + 0.5 * h * k1, u)
        k3 = state_derivative(ssm, x + 0.5 * h * k2, u)
        k4 = state_derivative(ssm, x + h * k3, u)
        x = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return x


def observation(ssm: SSM, x, u):
    ugv = ssm.ugv
    px, py, psi, vx, vy, r, delta, k_w = x[p.S.PX], x[p.S.PY], x[p.S.PSI], x[p.S.VX], x[p.S.VY], x[p.S.R], x[p.S.DELTA], x[p.S.KW]

    vydot = state_derivative(ssm, x, u)[p.S.VY]
    wheel_radius = ugv.R0 * k_w
    v_left, v_right = vx - r * ugv.tw / 2, vx + r * ugv.tw / 2
    side = vy + r * ugv.Lf
    y = np.zeros(p.NY)
    y[p.O.GPS_X], y[p.O.GPS_Y] = px, py
    y[p.O.GPS_SPEED] = np.hypot(vx, vy)
    y[p.O.GPS_COURSE] = psi + np.arctan2(vy, max(vx, p.V_EPS))
    y[p.O.IND_SPEED] = vx
    y[p.O.W_FL] = (v_left * np.cos(delta) + side * np.sin(delta)) / wheel_radius
    y[p.O.W_FR] = (v_right * np.cos(delta) + side * np.sin(delta)) / wheel_radius
    y[p.O.W_RL] = v_left / wheel_radius
    y[p.O.W_RR] = v_right / wheel_radius
    y[p.O.YAW_RATE] = r + x[p.S.BR]
    y[p.O.AX] = longitudinal_acceleration(ssm, x, u) + x[p.S.BAX]
    y[p.O.AY] = vydot + r * vx + x[p.S.BAY]
    y[p.O.STEER] = ugv.sr * delta

    return y
