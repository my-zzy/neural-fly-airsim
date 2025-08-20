#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data collection with user's tuned (Neural-Fly style) controller integrated.
Outputs schema identical to manual_wind_profiles:
t,p,p_d,v,v_d,q,R,w,T_sp,q_sp,hover_throttle,fa,pwm

- Uses figure-8 (fixed altitude) for test, random spline for training
- Wind profiles for train/test as specified
- fa excludes gravity: fa = m*a_world - R*[0,0,T]^T - m*g*e3, with T=(T_sp/HOVER_THROTTLE)*m*g
- Adds altitude floor protection to avoid ground impact
"""

import csv, time, json, math
from pathlib import Path
import numpy as np
import airsim

# ========= Constants / Config =========
DT = 0.02                         # 50 Hz
TRAIN_DURATION_S = 180            # 3 minutes per training profile
TEST_DURATION_S  = 60             # 1 minute per test profile

MASS_KG = 1.0
G = 9.81
HOVER_THROTTLE = 0.68             # set to your calibrated hover throttle
TAKEOFF_ALT = -5.0                # NED (Down positive, -5m above ground)

# Safety
SAFE_FLOOR = -1.0                 # never go above -1.0 (i.e., <1m above ground)
RECOVER_TO  = -2.2                # recover to -2.2m
T_SP_RECOVER = min(1.0, HOVER_THROTTLE + 0.25)

OUT_DIR_TRAIN = Path("logs_pid_random_profiles")
OUT_DIR_TEST  = Path("logs_pid_test_fig8")

# Training wind profiles (5 kinds, 3min each)
PROFILES_TRAIN = [
    {"tag":"0mps",   "kind":"const", "dir":(1,0,0), "mag":0.0},
    {"tag":"4.2mps", "kind":"const", "dir":(1,0,0), "mag":4.2},
    {"tag":"8.5mps", "kind":"const", "dir":(1,0,0), "mag":8.5},
    {"tag":"12.1mps","kind":"const", "dir":(1,0,0), "mag":12.1},
    {"tag":"sinusoidal_0to12mps","kind":"sin","dir":(1,0,0),"mag_mean":6.0,"mag_amp":6.0,"freq_hz":0.25}
]
# Test wind profiles (5 kinds, 1min each)
PROFILES_TEST = [
    {"tag":"2.0mps", "kind":"const", "dir":(1,0,0), "mag":2.0},
    {"tag":"6.0mps", "kind":"const", "dir":(1,0,0), "mag":6.0},
    {"tag":"10.0mps","kind":"const", "dir":(1,0,0), "mag":10.0},
    {"tag":"sinusoidal_0to8mps","kind":"sin","dir":(1,0,0),"mag_mean":4.0,"mag_amp":4.0,"freq_hz":0.33},
    {"tag":"gusty_12mps","kind":"gust","dir":(1,0,0),"mag":12.0,"noise_std":1.0}
]

# ========= Utilities =========
def v3r(v): return [float(v.x_val), float(v.y_val), float(v.z_val)]
def to_json_arr(x):
    if isinstance(x, (list, tuple, np.ndarray)):
        return json.dumps([float(v) for v in np.array(x).reshape(-1)])
    return json.dumps([float(x)])
def to_json_mat(M): return json.dumps(np.array(M, dtype=float).tolist())

def quat_wxyz_to_R(qw, qx, qy, qz):
    w,x,y,z = qw,qx,qy,qz
    return np.array([
        [1-2*(y*y+z*z),   2*(x*y - z*w),   2*(x*z + y*w)],
        [  2*(x*y + z*w), 1-2*(x*x+z*z),   2*(y*z - x*w)],
        [  2*(x*z - y*w),   2*(y*z + x*w), 1-2*(x*x+y*y)]
    ], dtype=float)

def quat_to_euler_xyz(qx, qy, qz, qw):
    sinr_cosp = 2*(qw*qx + qy*qz)
    cosr_cosp = 1 - 2*(qx*qx + qy*qy)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = 2*(qw*qy - qz*qx)
    if abs(sinp) >= 1: pitch = math.copysign(math.pi/2, sinp)
    else: pitch = math.asin(sinp)
    siny_cosp = 2*(qw*qz + qx*qy)
    cosy_cosp = 1 - 2*(qy*qy + qz*qz)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw

def euler_to_quat_xyz(roll, pitch, yaw):
    cr, sr = math.cos(roll*0.5), math.sin(roll*0.5)
    cp, sp = math.cos(pitch*0.5), math.sin(pitch*0.5)
    cy, sy = math.cos(yaw*0.5), math.sin(yaw*0.5)
    qw = cr*cp*cy + sr*sp*sy
    qx = sr*cp*cy - cr*sp*sy
    qy = cr*sp*cy + sr*cp*sy
    qz = cr*cp*sy - sr*sp*cy
    return qx, qy, qz, qw

def compute_fa_from_Tsp(m, a_world, R_b2w, T_sp):
    """fa = m*a_world - R_b2w*[0,0,T]^T - m*g*e3,  T=(T_sp/HOVER_THROTTLE)*m*g"""
    T = (T_sp / HOVER_THROTTLE) * m * G
    f_b = np.array([0.0, 0.0, T])       # body +Z (Down) thrust
    fT_world = R_b2w @ f_b
    e3 = np.array([0.0, 0.0, 1.0])
    return m*a_world - fT_world - m*G*e3

# ========= Trajectories with derivatives =========
def fig8_traj_with_deriv(t):
    """Figure-8 XY, fixed altitude z=-5m"""
    x  = 10.0 * math.sin(t * 0.5)
    y  = 10.0 * math.sin(t * 0.5) * math.cos(t * 0.5)
    z  = -5.0
    vx = 10.0 * 0.5 * math.cos(t * 0.5)
    vy = 10.0 * (0.5*math.cos(t*0.5)*math.cos(t*0.5) - 0.5*math.sin(t*0.5)*math.sin(t*0.5))
    vz = 0.0
    yaw = 0.0
    return (x,y,z,yaw), (vx,vy,vz)

def random_traj_with_deriv(t):
    """Random cubic-spline trajectory (180s), safe altitude [-6,-4]m"""
    import numpy as np
    from scipy.interpolate import CubicSpline
    if not hasattr(random_traj_with_deriv, "initialized"):
        np.random.seed(42)
        total_time = 180.0
        num_points = 8
        t_points = np.linspace(0, total_time, num_points)
        x_points = np.random.uniform(-10, 10, num_points)
        y_points = np.random.uniform(-10, 10, num_points)
        z_points = np.random.uniform(-6, -4, num_points)
        yaw_points = np.zeros(num_points)
        random_traj_with_deriv.total_time = total_time
        random_traj_with_deriv.x_spline = CubicSpline(t_points, x_points)
        random_traj_with_deriv.y_spline = CubicSpline(t_points, y_points)
        random_traj_with_deriv.z_spline = CubicSpline(t_points, z_points)
        random_traj_with_deriv.yaw_spline = CubicSpline(t_points, yaw_points)
        random_traj_with_deriv.initialized = True
    t = np.clip(t, 0, random_traj_with_deriv.total_time)
    x = float(random_traj_with_deriv.x_spline(t))
    y = float(random_traj_with_deriv.y_spline(t))
    z = float(random_traj_with_deriv.z_spline(t))
    yaw = float(random_traj_with_deriv.yaw_spline(t))
    vx = float(random_traj_with_deriv.x_spline.derivative()(t))
    vy = float(random_traj_with_deriv.y_spline.derivative()(t))
    vz = float(random_traj_with_deriv.z_spline.derivative()(t))
    return (x,y,z,yaw), (vx,vy,vz)

# ========= Wind helpers =========
def apply_wind_profile(client, prof, t_rel=None):
    d = np.array(prof["dir"], dtype=float); d = d/(np.linalg.norm(d)+1e-12)
    kind = prof["kind"]
    if kind == "const":
        mag = float(prof["mag"]); wvec = d*mag
    elif kind == "sin":
        mag = float(prof["mag_mean"]) + float(prof["mag_amp"])*math.sin(2*math.pi*float(prof["freq_hz"])*(t_rel or 0.0))
        mag = max(0.0, mag); wvec = d*mag
    elif kind == "gust":
        base = float(prof["mag"]); noise = float(prof.get("noise_std", 1.0))*np.random.randn(); wvec = d*(base + noise)
    else:
        wvec = np.zeros(3, dtype=float)
    client.simSetWind(airsim.Vector3r(*wvec.tolist()))
    return wvec

# ========= User's tuned controller (ported & minimal) =========
class TunedController:
    """
    Minimal wrapper of user's neural_fly_controller:
    - Keeps adaptive states a_hat, P
    - Produces (throttle, roll_des, pitch_des, yaw_des)
    Note: we keep a constant phi (as in user's test code) to avoid external deps.
    """
    def __init__(self, dt=DT):
        self.dt = dt
        # adaptive params
        self.h = 4
        self.a_hat = np.zeros(self.h, dtype=np.float64)
        self.P = np.eye(self.h, dtype=np.float64)
        # design params
        self.lambda_a = 0.1
        self.Q = np.eye(self.h, dtype=np.float64) * 0.01
        self.R = np.eye(3, dtype=np.float64) * 0.1
        self.K = np.eye(3, dtype=np.float64) * 5.0
        self.Lam = np.eye(3, dtype=np.float64) * 2.0
        # constant phi like user's test (3x4)
        self.phi_const = np.array([[0.1,0.2,-0.3,1.0],
                                   [0.1,0.2,-0.3,1.0],
                                   [0.1,0.2,-0.3,1.0]], dtype=np.float64)

    def step(self, client, pos, vel, att_euler, des_pos, des_vel, des_yaw):
        # state vectors
        p = np.asarray(pos, dtype=np.float64)
        v = np.asarray(vel, dtype=np.float64)
        roll, pitch, yaw = att_euler
        xd = np.asarray(des_pos, dtype=np.float64)
        xd_dot = np.asarray(des_vel, dtype=np.float64)

        # numerical desired acc ~ 0 for our analytic tracks (optional):
        xd_ddot = np.zeros(3, dtype=np.float64)

        # tracking surface
        q_tilde = p - xd
        s = v - xd_dot + (self.Lam @ q_tilde)

        # learning features (constant phi as user's test)
        phi = self.phi_const  # shape (3,h)
        y = np.zeros(3, dtype=np.float64)  # residual measurement (zero)

        # force command (m=1) in world
        f_nominal = xd_ddot + np.array([0.0, 0.0, 9.81])  # add gravity
        f_learning = phi @ self.a_hat
        u = f_nominal - (self.K @ s) - f_learning

        # adaptive updates (discretized)
        # P_phi_T @ inv(R) @ (phi @ a_hat - y)
        R_inv = np.linalg.inv(self.R)
        P_phi_T = self.P @ phi.T
        a_hat_dot = -self.lambda_a * self.a_hat - P_phi_T @ (R_inv @ (phi @ self.a_hat - y)) + P_phi_T @ s
        self.a_hat = self.a_hat + a_hat_dot * self.dt

        P_dot = -2*self.lambda_a*self.P + self.Q - P_phi_T @ (R_inv @ (phi @ self.P))
        self.P = self.P + P_dot * self.dt

        # map force -> attitude/throttle (simplified small-angle)
        thrust_mag = np.linalg.norm(u)
        if thrust_mag > 1e-6:
            u_norm = u / thrust_mag
            accel_x = u_norm[0]
            accel_y = -u_norm[1]            # note the minus (user's code)
            psi = des_yaw
            roll_des  = -(accel_y * math.cos(psi) - accel_x * math.sin(psi))
            pitch_des =  (accel_x * math.cos(psi) + accel_y * math.sin(psi))
            # limit
            max_ang = math.radians(30.0)
            roll_des  = float(np.clip(roll_des, -max_ang, max_ang))
            pitch_des = float(np.clip(pitch_des, -max_ang, max_ang))
            yaw_des   = des_yaw
        else:
            roll_des = pitch_des = 0.0; yaw_des = des_yaw

        # crude throttle mapping from u_z -> [0,1]
        # keep consistent with fa mapping (we only need T_sp number)
        # here just bias around hover and add proportional to -u_z (world Down)
        th = HOVER_THROTTLE + 0.03 * (-u[2])
        T_sp = float(np.clip(th, 0.0, 1.0))

        return T_sp, roll_des, pitch_des, yaw_des, u

# ========= Core collection =========
def reset_and_takeoff(client):
    client.reset(); time.sleep(1.0)
    client.enableApiControl(True); client.armDisarm(True)
    client.takeoffAsync().join()
    client.moveToZAsync(TAKEOFF_ALT, 1.0).join()

def collect_one_profile(client, out_csv_path, duration_s, prof, traj_func):
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)
    ctrl = TunedController(dt=DT)

    with open(out_csv_path, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["t","p","p_d","v","v_d","q","R","w","T_sp","q_sp","hover_throttle","fa","pwm"])

        v_prev = None
        t0 = time.time()
        while True:
            now = time.time()
            t = now - t0
            if t >= duration_s: break

            # set wind
            apply_wind_profile(client, prof, t_rel=t if prof["kind"]!="const" else None)

            # read state
            st = client.getMultirotorState()
            kin = st.kinematics_estimated
            p = np.asarray(v3r(kin.position), dtype=float)
            v = np.asarray(v3r(kin.linear_velocity), dtype=float)
            omg = np.asarray(v3r(kin.angular_velocity), dtype=float)
            q = kin.orientation
            qw,qx,qy,qz = q.w_val,q.x_val,q.y_val,q.z_val
            R = quat_wxyz_to_R(qw,qx,qy,qz)
            roll,pitch,yaw = quat_to_euler_xyz(qx,qy,qz,qw)

            # desired
            (xd,yd,zd,psid), (vxd,vyd,vzd) = traj_func(t)
            p_d = np.array([xd,yd,zd], dtype=float)
            v_d = np.array([vxd,vyd,vzd], dtype=float)

            # floor protection
            if p[2] > SAFE_FLOOR:
                client.moveByRollPitchYawThrottleAsync(0.0, 0.0, yaw, T_SP_RECOVER, duration=DT)
                # also nudge altitude if needed
                if p[2] > RECOVER_TO:
                    client.moveToZAsync(RECOVER_TO, 1.0).join()
                # log anyway with current state
                a_world = np.zeros(3) if v_prev is None else (v - v_prev)/DT
                v_prev = v.copy()
                fa = compute_fa_from_Tsp(MASS_KG, a_world, R, T_SP_RECOVER)
                wr.writerow([
                    f"{t:.14g}", to_json_arr(p), to_json_arr(p_d),
                    to_json_arr(v), to_json_arr(v_d),
                    to_json_arr([qx,qy,qz,qw]), to_json_mat(R), to_json_arr(omg),
                    to_json_arr([T_SP_RECOVER]),
                    to_json_arr(euler_to_quat_xyz(0.0,0.0,yaw)),
                    to_json_arr([HOVER_THROTTLE]), to_json_arr(fa), to_json_arr([0,0,0,0])
                ])
                # loop rate
                elapsed = time.time() - now
                if elapsed < DT: time.sleep(DT - elapsed)
                continue

            # controller
            T_sp, r_des, p_des, yaw_des, u = ctrl.step(client, p, v, (roll,pitch,yaw), p_d, v_d, psid)

            # send command
            client.moveByRollPitchYawThrottleAsync(r_des, p_des, yaw_des, T_sp, duration=DT)

            # accel
            if v_prev is None: a_world = np.zeros(3, dtype=float)
            else: a_world = (v - v_prev)/DT
            v_prev = v.copy()

            # fa
            fa = compute_fa_from_Tsp(MASS_KG, a_world, R, T_sp)

            # motor speeds as pwm proxy
            pwm_list = []
            try:
                rotor_states = client.getRotorStates()
                for i in range(4):
                    r = rotor_states.rotors[i]
                    spd = r['speed'] if isinstance(r, dict) else getattr(r, "speed", 0.0)
                    pwm_list.append(float(spd))
            except Exception:
                pwm_list = [0.0,0.0,0.0,0.0]

            # desired attitude quaternion
            qx_sp,qy_sp,qz_sp,qw_sp = euler_to_quat_xyz(r_des, p_des, yaw_des)

            # log
            wr.writerow([
                f"{t:.14g}",
                to_json_arr(p),
                to_json_arr(p_d),
                to_json_arr(v),
                to_json_arr(v_d),
                to_json_arr([qx,qy,qz,qw]),
                to_json_mat(R),
                to_json_arr(omg),
                to_json_arr([T_sp]),
                to_json_arr([qx_sp,qy_sp,qz_sp,qw_sp]),
                to_json_arr([HOVER_THROTTLE]),
                to_json_arr(fa),
                to_json_arr(pwm_list),
            ])

            # loop rate
            elapsed = time.time() - now
            if elapsed < DT: time.sleep(DT - elapsed)

    print(f"[OK] Saved: {out_csv_path}")

# ========= Main =========
def reset_and_takeoff(client):
    client.reset(); time.sleep(1.0)
    client.enableApiControl(True); client.armDisarm(True)
    client.takeoffAsync().join()
    client.moveToZAsync(TAKEOFF_ALT, 1.0).join()

def main():
    client = airsim.MultirotorClient(); client.confirmConnection()

    # Train: random spline
    for i, prof in enumerate(PROFILES_TRAIN, 1):
        print(f"\n[Train {i}/{len(PROFILES_TRAIN)}] profile={prof['tag']} ... reset & takeoff")
        reset_and_takeoff(client)
        out_csv = OUT_DIR_TRAIN / f"pid_random_{prof['tag']}.csv"
        collect_one_profile(client, out_csv, TRAIN_DURATION_S, prof, random_traj_with_deriv)

    # Test: figure-8
    for i, prof in enumerate(PROFILES_TEST, 1):
        print(f"\n[Test {i}/{len(PROFILES_TEST)}] profile={prof['tag']} ... reset & takeoff")
        reset_and_takeoff(client)
        out_csv = OUT_DIR_TEST / f"pid_fig8_{prof['tag']}.csv"
        collect_one_profile(client, out_csv, TEST_DURATION_S, prof, fig8_traj_with_deriv)

    # cleanup
    client.armDisarm(False)
    client.enableApiControl(False)
    print("\nAll collections finished.")

if __name__ == "__main__":
    main()
