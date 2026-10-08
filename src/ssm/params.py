STATE_VARS_ = [
    "px", "py", "psi", 
    "vx", "vy", "r", 
    "b_ax", "b_ay", "b_r", 
    "k_w", "delta"
]

INPUT_VARS_ = [
    "pedal", "brake", "ratio"
]

OBSERVATTIONS_ = [
    "gps_x", "gps_y", "gps_speed",
    "gps_course", "ind_speed", "w_fl", 
    "w_fr", "w_rl", "w_rr", "yaw_rate", 
    "ax", "ay", "steer"
]

NX_ = len(STATE_VARS_)
NU_ = len(INPUT_VARS_)
NY_ = len(OBSERVATTIONS_)

class S_:
    PX, PY, PSI, VX, VY, R, BAX, BAY, BR, KW, DELTA = range(NX_)
class I_: 
    PEDAL, BRAKE, RATIO = range(NU_)
class O_: 
    GPS_X, GPS_Y, GPS_SPEED, GPS_COURSE, IND_SPEED, W_FL, W_FR, W_RL, W_RR, YAW_RATE, AX, AY, STEER = range(NY_)


PROCESS_NOISE_ = {
    "px": 1e-4, "py": 1e-4, "psi": 1e-5, 
    "vx": 0.5, "vy": 0.1, "r": 0.05,
    "b_ax": 1e-5, "b_ay": 1e-5, "b_r": 1e-7, 
    "k_w": 1e-7, "delta": 0.02
}

SENSOR_BIAS_ = {
    "gps_x": 1.0, "gps_y": 1.0, "gps_speed": 0.15, 
    "gps_course": 0.08, "ind_speed": 0.3,
    "w_fl": 0.3, "w_fr": 0.3, "w_rl": 0.3, 
    "w_rr": 0.3, "yaw_rate": 0.05, "ax": 0.3, 
    "ay": 0.3, "steer": 0.05
}

V_EPS_ = 0.1

class PARAMS:
    STATE_VARS = STATE_VARS_
    INPUT_VARS = INPUT_VARS_
    OBSERVATTIONS = OBSERVATTIONS_
    S = S_
    I = I_
    O = O_
    PROCESS_NOISE = PROCESS_NOISE_
    SENSOR_BIAS = SENSOR_BIAS_
    NX = NX_
    NU = NU_
    NY = NY_
    V_EPS = V_EPS_

