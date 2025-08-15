#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import math
import time
import argparse
import numpy as np
import torch
import airsim
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# === 从config.py 读取飞控外环与质量等参数 ===
from config import (
    UAV_mass,            # kg
    cx, cy, cz,          # 外环 P
    cu, cv, cw,          # 外环 D
    lamx, lamy, lamz     # 扰动估计自适应律增益
)

# ============================================================
# 安全常量（可按需调整）
# ============================================================
MAX_ACCEL = 4.0            # 水平轴向期望加速度限幅 [m/s^2]
MAX_TILT_DEG = 25.0        # 姿态限幅 [deg]
MIN_THRUST_G = 0.45        # 推力下限（相对 mg）
MAX_THRUST_G = 1.40        # 推力上限（相对 mg）
DENOM_FLOOR = 0.60         # 防奇异: cosφ cosθ 的最小值（过小会让总推力爆表）
Y_SIGN = 1.0              # AirSim/NED 与常见推导在 y 轴方向的符号修正
AZ_MAX_UP    = 3.0       # 向上最大 ẇ_cmd（m/s^2，负号是向上，加在 clip 里）
AZ_MAX_DOWN  = 2.5       # 向下最大 ẇ_cmd（m/s^2）
HOVER_THR    = 0.68      # 从日志估出来的悬停节流（原先硬写 0.5 明显偏低）
# Yaw 微调
K_YAW_P = 0.8
K_YAW_D = 0.05             # 使用机体系 z 角速度（rad/s）

# 姿态一阶低通（抑制抖动）
LPF_COEF = 0.75            # 0.7~0.9 建议

# ============================================================
# 旋转/角度工具
# ============================================================
def quaternion_to_euler(x, y, z, w):
    # roll
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    # pitch
    sinp = 2 * (w * y - z * x)
    pitch = math.copysign(math.pi / 2, sinp) if abs(sinp) >= 1 else math.asin(sinp)
    # yaw
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw

def euler_to_Rbn(roll, pitch, yaw):
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    # body -> NED
    R = np.array([
        [ cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr ],
        [ sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr ],
        [  -sp ,        cp*sr    ,        cp*cr    ]
    ], dtype=np.float32)
    return R

def euler_to_Rnb(roll, pitch, yaw):
    return euler_to_Rbn(roll, pitch, yaw).T  # NED -> body

def euler_to_quat(roll, pitch, yaw):
    cy, sy = math.cos(yaw*0.5), math.sin(yaw*0.5)
    cp, sp = math.cos(pitch*0.5), math.sin(pitch*0.5)
    cr, sr = math.cos(roll*0.5),  math.sin(roll*0.5)
    qw = cr*cp*cy + sr*sp*sy
    qx = sr*cp*cy - cr*sp*sy
    qy = cr*sp*cy + sr*cp*sy
    qz = cr*cp*sy - sr*sp*cy
    return qx, qy, qz, qw

def wrap_pi(a: float) -> float:
    """wrap angle to [-pi, pi]."""
    return (a + math.pi) % (2*math.pi) - math.pi

# ============================================================
# Meta-PINN 残差力前馈（可选）
# ============================================================
class ForceCompensator:
    """
    加载 Meta-PINN，并提供 body-frame 残差力预测接口（无梯度推理）。
    """
    def __init__(self, ckpt_path, scaler_npz, *, features, num_tasks, task_id,
                 device="cpu", cond_dim=1, use_cond_mod=True,
                 task_dim=128, hidden_dim=384, beta_min=0.15, beta_max=8.0):
        self.device = torch.device(device)
        self.features = list(features)
        self.task_id = int(task_id)
        self.cond_dim = int(cond_dim)

        # 计算输入维度（与训练保持一致）
        feat_dim = 0
        for f in self.features:
            if f == "v":   feat_dim += 3
            elif f == "q": feat_dim += 4
            elif f == "rpy": feat_dim += 3
            elif f == "pwm": feat_dim += 4
            else: raise ValueError(f"Unknown feature '{f}'")
        input_dim = feat_dim

        # 构建模型
        try:
            from meta_pinn import MetaPINN
        except Exception:
            from meta_pinn.model import MetaPINN  # 兼容你的项目结构

        self.model = MetaPINN(
            input_dim=input_dim,
            num_tasks=num_tasks,
            task_dim=task_dim,
            hidden_dim=hidden_dim,
            use_uncertainty=True,
            cond_dim=self.cond_dim,
            use_cond_mod=use_cond_mod,
            cond_mod_from='target',
            beta_min=beta_min, beta_max=beta_max,
        ).to(self.device).eval()

        # 加载权重
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        try:
            sd = torch.load(ckpt_path, map_location=self.device)
            self.model.load_state_dict(sd)
        except RuntimeError as e:
            raise RuntimeError(
                f"[Meta-PINN] load_state_dict 失败：{e}\n"
                f"很可能是 --num_tasks 与训练时不一致。请把 --num_tasks 设置为训练用的任务数。"
            )

        # 输入标准化
        if scaler_npz and os.path.isfile(scaler_npz):
            d = np.load(scaler_npz)
            self.x_mean = torch.tensor(d["x_mean"], dtype=torch.float32, device=self.device).view(1, -1)
            self.x_std  = torch.tensor(d["x_std"],  dtype=torch.float32, device=self.device).view(1, -1)
        else:
            self.x_mean = torch.zeros(1, input_dim, dtype=torch.float32, device=self.device)
            self.x_std  = torch.ones(1,  input_dim, dtype=torch.float32, device=self.device)

    def _make_x(self, v_ned, rpy, throttle):
        roll, pitch, yaw = rpy
        R_nb = euler_to_Rnb(roll, pitch, yaw)  # NED->body
        v_body = (R_nb @ np.asarray(v_ned, dtype=np.float32)).astype(np.float32)

        qx, qy, qz, qw = euler_to_quat(roll, pitch, yaw)
        pwm4 = np.array([throttle, throttle, throttle, throttle], dtype=np.float32)

        xs = []
        for f in self.features:
            if f == "v":   xs.append(v_body)
            elif f == "q": xs.append(np.array([qx, qy, qz, qw], dtype=np.float32))
            elif f == "rpy": xs.append(np.array([roll, pitch, yaw], dtype=np.float32))
            elif f == "pwm": xs.append(pwm4)
        x = np.concatenate(xs, axis=0).astype(np.float32)
        return x

    @torch.no_grad()
    def predict_body_force(self, v_ned, rpy, throttle, wind=0.0):
        x = self._make_x(v_ned, rpy, throttle)
        x_t = torch.from_numpy(x).to(self.device).view(1, -1)
        x_t = (x_t - self.x_mean) / (self.x_std + 1e-8)

        c = torch.zeros(1, self.cond_dim, dtype=torch.float32, device=self.device)
        if self.cond_dim >= 1:
            c[0, 0] = float(wind)

        f_body = self.model._pred_physical(x_t, task_id=self.task_id, c_in=c)  # [1,3]
        return f_body.squeeze(0).cpu().numpy()

# ============================================================
# 自适应控制（融合 Meta-PINN 前馈）
# ============================================================
def adaptive_controller(pos, vel, att, ang_vel, posd, attd,
                        dhat, jifen, dt, t,
                        *,
                        force_comp=None,
                        alpha=0.5,
                        prev_throttle=0.5,
                        wind_speed=0.0):
    # ---- 当前状态与参考 ----
    x, y, z = pos[0][-1], pos[1][-1], pos[2][-1]
    u, v, w = vel[0][-1], vel[1][-1], vel[2][-1]
    phi, theta, psi = att[0][-1], att[1][-1], att[2][-1]
    xd, yd, zd = posd[0][-1], posd[1][-1], posd[2][-1]
    psid = attd[2][-1]
    dx_hat, dy_hat, dz_hat, dphi_hat, dtheta_hat, dpsi_hat = dhat
    xphi, xtheta, xpsi = jifen
    g = 9.81

    # 数值导数
    def d1(a):  return (a[-1]-a[-2])/dt if len(a) >= 2 else 0.0
    def d2(a):  return (((a[-1]-a[-2])/dt) - ((a[-2]-a[-3])/dt))/dt if len(a) >= 3 else 0.0
    xd_dot,  yd_dot,  zd_dot  = d1(posd[0]), d1(posd[1]), d1(posd[2])
    xd_ddot, yd_ddot, zd_ddot = d2(posd[0]), d2(posd[1]), d2(posd[2])

    # ---- 外环误差动力学 ----
    # Z
    ez   = z - zd
    ew   = w - zd_dot + cz*ez
    ez_d = ew - cz*ez
    w_dot_cmd   = -cw*ew - ez + zd_ddot - cz*ez_d
    w_dot_cmd = float(np.clip(w_dot_cmd, -AZ_MAX_UP, AZ_MAX_DOWN))

    #dz_hat += lamz*ew * dt
    dz_hat_candidate = dz_hat + lamz*ew*dt
    T_bz_unsat = UAV_mass * (g + dz_hat_candidate - w_dot_cmd)
    denom = max(DENOM_FLOOR, math.cos(phi)*math.cos(theta))
    T_total_unsat = T_bz_unsat / denom
    T_total_clamped = float(np.clip(
        T_total_unsat, MIN_THRUST_G*UAV_mass*g, MAX_THRUST_G*UAV_mass*g
    ))
    # 如果确实触底，则丢弃这步积分，防止“越积越狠”
    if T_total_clamped <= MIN_THRUST_G*UAV_mass*g + 1e-6:
        dz_hat = dz_hat
    else:
        dz_hat = dz_hat_candidate


    # X
    ex   = x - xd
    eu   = u - xd_dot + cx*ex
    ex_d = eu - cx*ex
    u_dot_cmd   = -cu*eu - ex + xd_ddot - cx*ex_d
    dx_hat     += lamx*eu * dt
    ax_cmd = u_dot_cmd - dx_hat

    # Y
    ey   = y - yd
    ev   = v - yd_dot + cy*ey
    ey_d = ev - cy*ey
    v_dot_cmd   = -cv*ev - ey + yd_ddot - cy*ey_d
    dy_hat     += lamy*ev * dt
    ay_cmd = v_dot_cmd - dy_hat

    # ---- Meta-PINN 前馈（可选）----
    if force_comp is not None and alpha > 0.0:
        try:
            f_body = force_comp.predict_body_force(
                v_ned=[u, v, w],
                rpy=(phi, theta, psi),
                throttle=prev_throttle,
                wind=wind_speed
            )
            f_body = np.clip(f_body, -20.0, 20.0)
            R_bn   = euler_to_Rbn(phi, theta, psi)   # body->NED
            a_dist = (R_bn @ (f_body / UAV_mass)).astype(np.float32)
            # 融合到扰动估计
            dx_hat = (1-alpha)*dx_hat + alpha*a_dist[0]
            dy_hat = (1-alpha)*dy_hat + alpha*a_dist[1]
            dz_hat = (1-alpha)*dz_hat + alpha*a_dist[2]
            ax_cmd = u_dot_cmd - dx_hat
            ay_cmd = v_dot_cmd - dy_hat
        except Exception as e:
            print(f"[WARN] force_comp forward failed: {e}")

    # ---- 油门映射（防奇异 + 夹紧）----
    # m ẇ = -T_bz + mg + d_z  ->  T_bz = m(g + d_z - ẇ_cmd)
    T_bz = UAV_mass * (g + dz_hat - w_dot_cmd)
    denom = max(DENOM_FLOOR, math.cos(phi)*math.cos(theta))
    T_total = T_bz / denom
    T_total = float(np.clip(T_total, MIN_THRUST_G*UAV_mass*g, MAX_THRUST_G*UAV_mass*g))

    # AirSim 节流（约 0.5 为悬停）
    #throttle = (T_total / (UAV_mass*9.81)) * 0.5
    throttle = HOVER_THR * (T_total / (UAV_mass*9.81))
    throttle = float(np.clip(throttle, 0.0, 1.0))

    # ---- 水平加速度 -> 姿态 ----
    ay_cmd *= Y_SIGN
    ax_cmd = float(np.clip(ax_cmd, -MAX_ACCEL, MAX_ACCEL))
    ay_cmd = float(np.clip(ay_cmd, -MAX_ACCEL, MAX_ACCEL))

    roll_cmd  = ( ay_cmd*math.cos(psi) - ax_cmd*math.sin(psi) ) / g
    pitch_cmd = ( ax_cmd*math.cos(psi) + ay_cmd*math.sin(psi) ) / g

    max_ang = math.radians(MAX_TILT_DEG)
    roll_cmd  = float(np.clip(roll_cmd,  -max_ang, max_ang))
    pitch_cmd = float(np.clip(pitch_cmd, -max_ang, max_ang))


    # ---- yaw 微调 ----
    # 取机体系 z 角速度的最近值（标量），而不是整段历史列表
    wz = float(ang_vel[2][-1]) if len(ang_vel[2]) > 0 else 0.0

    yaw_err = wrap_pi(psid - psi)
    yaw_cmd = psi + K_YAW_P * yaw_err - K_YAW_D * wz

    dhat_new  = [dx_hat, dy_hat, dz_hat, dphi_hat, dtheta_hat, dpsi_hat]
    jifen_new = [xphi,  xtheta,  xpsi]
    return throttle, roll_cmd, pitch_cmd, yaw_cmd, dhat_new, jifen_new

# ============================================================
# 控制器封装
# ============================================================
class AirSimAdaptiveController:
    def __init__(self, force_comp=None, alpha=0.0, wind_speed=0.0, dt=0.01):
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)

        self.dt = float(dt)
        self.simulation_time = 0.0
        self.last_throttle = 0.5

        self.force_comp = force_comp
        self.alpha = float(alpha)
        self.wind_speed = float(wind_speed)

        # 自适应/积分
        self.dhat = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.jifen = [0.0, 0.0, 0.0]

        # 历史缓存
        self.pos_history = [[], [], []]
        self.vel_history = [[], [], []]
        self.att_history = [[], [], []]
        self.ang_vel_history = [[], [], []]
        self.posd_history = [[], [], []]
        self.attd_history = [[], [], []]

        # 姿态低通
        self._r_cmd_filt = 0.0
        self._p_cmd_filt = 0.0

        # 日志
        self.time_log = []
        self.position_log = []
        self.attitude_log = []
        self.control_log = []
        self.desired_pos_log = []
        self.desired_att_log = []

        print("AirSim Adaptive Controller initialized (Meta-PINN fusion: {})"
              .format("ON" if force_comp is not None and self.alpha > 0 else "OFF"))

    def get_state(self):
        state = self.client.getMultirotorState()
        p = state.kinematics_estimated.position
        v = state.kinematics_estimated.linear_velocity
        o = state.kinematics_estimated.orientation
        w = state.kinematics_estimated.angular_velocity

        position = [p.x_val, p.y_val, p.z_val]
        velocity = [v.x_val, v.y_val, v.z_val]
        roll, pitch, yaw = quaternion_to_euler(o.x_val, o.y_val, o.z_val, o.w_val)
        attitude = [roll, pitch, yaw]
        ang_vel = [w.x_val, w.y_val, w.z_val]
        return position, velocity, attitude, ang_vel

    def update_history(self, pos, vel, att, ang_vel, posd, attd):
        for i in range(3):
            self.pos_history[i].append(pos[i])
            self.vel_history[i].append(vel[i])
            self.att_history[i].append(att[i])
            self.ang_vel_history[i].append(ang_vel[i])
            self.posd_history[i].append(posd[i])
            self.attd_history[i].append(attd[i])
        max_history = 10
        for i in range(3):
            for buf in (self.pos_history, self.vel_history, self.att_history,
                        self.ang_vel_history, self.posd_history, self.attd_history):
                if len(buf[i]) > max_history:
                    buf[i][:] = buf[i][-max_history:]

    def send_control_to_airsim(self, throttle, roll_desired, pitch_desired, yaw_desired):
        try:
            self.client.moveByRollPitchYawThrottleAsync(
                roll_desired, pitch_desired, yaw_desired, throttle, duration=self.dt
            )
        except Exception as e:
            print(f"[ERR] send control failed: {e}")

    def run_simulation(self, total_time=20.0, trajectory_func=None):
        if trajectory_func is None:
            trajectory_func = test2
        print(f"Starting simulation for {total_time}s ... takeoff")
        self.client.takeoffAsync().join()
        time.sleep(1.0)

        step = 0
        while self.simulation_time < total_time:
            t0 = time.time()

            pos, vel, att, ang_vel = self.get_state()
            xd, yd, zd, psid = trajectory_func(self.simulation_time)
            desired_pos = [xd, yd, zd]
            desired_att = [0.0, 0.0, psid]

            self.update_history(pos, vel, att, ang_vel, desired_pos, desired_att)

            if len(self.pos_history[0]) >= 3:
                thr, r_des, p_des, y_des, self.dhat, self.jifen = adaptive_controller(
                    self.pos_history, self.vel_history, self.att_history, self.ang_vel_history,
                    self.posd_history, self.attd_history,
                    self.dhat, self.jifen, self.dt, self.simulation_time,
                    force_comp=self.force_comp,
                    alpha=self.alpha,
                    prev_throttle=self.last_throttle,
                    wind_speed=self.wind_speed
                )
                self.last_throttle = thr

                # 姿态低通
                self._r_cmd_filt = LPF_COEF*self._r_cmd_filt + (1.0-LPF_COEF)*r_des
                self._p_cmd_filt = LPF_COEF*self._p_cmd_filt + (1.0-LPF_COEF)*p_des
                r_out = self._r_cmd_filt
                p_out = self._p_cmd_filt

                self.send_control_to_airsim(thr, r_out, p_out, y_des)
                desired_att = [r_out, -p_out, y_des]

                # log
                self.time_log.append(self.simulation_time)
                self.position_log.append(pos.copy())
                self.attitude_log.append(att.copy())
                self.control_log.append([thr, r_out, p_out, y_des])
                self.desired_pos_log.append(desired_pos.copy())
                self.desired_att_log.append(desired_att.copy())

                if step % int(max(1, 1.0/self.dt)) == 0:
                    print(f"t={self.simulation_time:5.2f}s | pos=({pos[0]:6.2f},{pos[1]:6.2f},{pos[2]:6.2f}) "
                          f"| vel=({vel[0]:5.2f},{vel[1]:5.2f},{vel[2]:5.2f}) "
                          f"| thr={thr:.3f} rpy=({math.degrees(r_out):.1f},{math.degrees(p_out):.1f},{math.degrees(y_des):.1f})")

            step += 1
            self.simulation_time += self.dt
            time.sleep(max(0.0, self.dt - (time.time() - t0)))

        print("Landing ...")
        self.client.landAsync().join()
        self.client.armDisarm(False)
        self.client.enableApiControl(False)
        print("Simulation done.")
        return self.get_logged_data()

    def get_logged_data(self):
        return {
            'time': np.array(self.time_log),
            'position': np.array(self.position_log),
            'attitude': np.array(self.attitude_log),
            'control': np.array(self.control_log),
            'desired_position': np.array(self.desired_pos_log),
            'desired_attitude': np.array(self.desired_att_log)
        }

    def plot_results(self, data):
        if data['time'].size == 0:
            print("No data to plot.")
            return

        # 位置/姿态跟踪
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        for i, lab in enumerate(['X','Y','Z']):
            axes[0,i].plot(data['time'], data['position'][:,i], 'b-', label='Actual', linewidth=2)
            axes[0,i].plot(data['time'], data['desired_position'][:,i], 'r--', label='Desired', linewidth=2)
            axes[0,i].set_title(f'{lab}-Position'); axes[0,i].grid(True); axes[0,i].legend()
        for i, lab in enumerate(['Roll','Pitch','Yaw']):
            axes[1,i].plot(data['time'], np.degrees(data['attitude'][:,i]), 'b-', label='Actual', linewidth=2)
            axes[1,i].plot(data['time'], np.degrees(data['desired_attitude'][:,i]), 'r--', label='Desired', linewidth=2)
            axes[1,i].set_title(f'{lab} Tracking'); axes[1,i].grid(True); axes[1,i].legend()
        plt.tight_layout(); plt.show()

        # 3D 轨迹
        # fig = plt.figure(figsize=(12,8))
        # ax = fig.add_subplot(111, projection='3d')
        # ax.plot(data['position'][:,0], data['position'][:,1], data['position'][:,2], 'b-', label='Actual', linewidth=2)
        # ax.plot(data['desired_position'][:,0], data['desired_position'][:,1], data['desired_position'][:,2], 'r--', label='Desired', linewidth=2)
        # ax.legend(); ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z'); ax.grid(True); plt.tight_layout(); plt.show()

        # 2D 轨迹 (X-Y)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(data['position'][:, 0], data['position'][:, 1], 'b-', label='Actual', linewidth=2)
        ax.plot(data['desired_position'][:, 0], data['desired_position'][:, 1], 'r--', label='Desired', linewidth=2)
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_title('2D Trajectory (X-Y)')
        ax.legend(); ax.grid(True); plt.tight_layout(); plt.show()

        # 控制量
        fig, axes = plt.subplots(2,2, figsize=(12,8))
        labels = ['Throttle','Roll des (deg)','Pitch des (deg)','Yaw des (deg)']
        for i in range(4):
            r, c = i//2, i%2
            y = data['control'][:,i] if i==0 else np.degrees(data['control'][:,i])
            axes[r,c].plot(data['time'], y, 'g-'); axes[r,c].set_title(labels[i]); axes[r,c].grid(True)
        # plt.tight_layout(); plt.show()

# ============================================================
# 轨迹样例
# ============================================================
def test1(t):
    return 0.1*t, 0.0, -5.0 - 1.0*t, 0.0

def test2(t):
    x = 10.0 * math.sin(t * 0.5)
    y = 10.0 * math.sin(t * 0.5) * math.cos(t * 0.5)
    z = -10.0 - 2.0*t
    z = -5.0
    yaw = 0.0
    return x, y, z, yaw

# ============================================================
# Main
# ============================================================
def parse_args():
    p = argparse.ArgumentParser("AirSim + Meta-PINN fused adaptive control (no-grad)")
    p.add_argument("--ckpt", type=str, default="saved_models/meta_pinn/meta_pinn_last.pth",
                   help="Meta-PINN checkpoint 路径")
    p.add_argument("--scaler_npz", type=str, default="saved_models/meta_pinn/x_scaler.npz",
                   help="输入标准化 npz（含 x_mean/x_std）")
    p.add_argument("--features", type=str, default="v,q,pwm",
                   help="与训练一致的特征：如 v,q,pwm 或 v,rpy,pwm（逗号分隔）")
    p.add_argument("--num_tasks", type=int, default=6, help="训练时的任务数（要与 ckpt 一致）")
    p.add_argument("--task_id", type=int, default=3, help="使用哪个任务嵌入")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--alpha", type=float, default=1, help="Meta-PINN 残差前馈权重 [0,1]；0 为关闭")
    p.add_argument("--wind", type=float, default=0.0, help="条件向量（如风速），需与训练 cond_dim 对齐")
    p.add_argument("--sim_time", type=float, default=20.0)
    p.add_argument("--dt", type=float, default=0.01)
    p.add_argument("--traj", type=str, default="test2", choices=["test1","test2"])
    return p.parse_args()

def main():
    args = parse_args()

    # 构建可选的 Meta-PINN 前馈
    features = [s.strip() for s in args.features.split(",") if s.strip()]
    force_comp = None
    if args.alpha > 0.0:
        force_comp = ForceCompensator(
            ckpt_path=args.ckpt,
            scaler_npz=args.scaler_npz,
            features=features,
            num_tasks=args.num_tasks,
            task_id=args.task_id,
            device=args.device,
            cond_dim=1, use_cond_mod=True,
            task_dim=128, hidden_dim=384,
            beta_min=0.15, beta_max=8.0
        )

    controller = AirSimAdaptiveController(
        force_comp=force_comp,
        alpha=args.alpha,
        wind_speed=args.wind,
        dt=args.dt
    )
    traj_fn = test1 if args.traj == "test1" else test2

    data = controller.run_simulation(total_time=args.sim_time, trajectory_func=traj_fn)
    controller.plot_results(data)

if __name__ == "__main__":
    main()
