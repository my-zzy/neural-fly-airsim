#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AirSim + 自适应控制 + 线上 Meta-PINN（离线权重）
"""
import os, time, math, argparse, csv
import numpy as np
import torch
import torch.nn.functional as F
import airsim
import matplotlib.pyplot as plt
from pathlib import Path
from collections import deque

from config import *  # UAV_mass, cx,cy,cz, cu,cv,cw, lamx,lamy,lamz
from meta_pinn.model import MetaPINN  

HOVER_THROTTLE = 0.594
G = 9.81
RESULTS_DIR = Path("result")

def clip(x, lo, hi): return lo if x < lo else (hi if x > hi else x)

# ---------------- 风况全集 ----------------
PROFILES_ALL = {
    "0mps":  {"tag":"0mps",  "kind":"const", "dir":(0,1,0), "mag":0.0},
    "5mps":{"tag":"5mps","kind":"const", "dir":(0,1,0), "mag":5.0},
    "10mps":{"tag":"10mps","kind":"const", "dir":(0,1,0), "mag":10.0},
    "12mps":{"tag":"12mps","kind":"const", "dir":(0,1,0), "mag":12.0},
    "13.5mps":{"tag":"13.5mps","kind":"const", "dir":(0,1,0), "mag":13.5},
    "15mps":{"tag":"15mps","kind":"const", "dir":(0,1,0), "mag":15.0},
    "sinusoidal_0to10mps":  {"tag":"sinusoidal_0to10mps","kind":"sin","dir":(0,1,0),"mag_mean":5.0,"mag_amp":5.0,"freq_hz":0.33},
    "sinusoidal_0to18mps": {"tag":"sinusoidal_0to18mps","kind":"sin","dir":(0,1,0),"mag_mean":9.0,"mag_amp":9.0,"freq_hz":0.25}
}

WIND_CONDITIONS = {
    "0mps": "nowind",
    "5mps": "5wind",
    "10mps": "10wind",
    "12mps": "12wind",
    "13.5mps": "13p5wind",
    "15mps": "15wind",
    "sinusoidal_0to10mps": "10sint",
    "sinusoidal_0to18mps": "18sint"
}
TRAIN_CONDS = ["nowind", "5wind", "10wind", "12wind", "13p5wind","10sint"]
TEST_CONDS  = ["nowind", "5wind", "10wind", "12wind", "13p5wind", "15wind", "18sint"]

# ---------------- 姿态/旋转工具 ----------------
def quaternion_to_euler(x, y, z, w):
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = 2 * (w * y - z * x)
    pitch = math.copysign(math.pi/2, sinp) if abs(sinp) >= 1 else math.asin(sinp)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw

def euler_to_quat(roll, pitch, yaw):
    cr, sr = math.cos(roll/2), math.sin(roll/2)
    cp, sp = math.cos(pitch/2), math.sin(pitch/2)
    cy, sy = math.cos(yaw/2), math.sin(yaw/2)
    qw = cr*cp*cy + sr*sp*sy
    qx = sr*cp*cy - cr*sp*sy
    qy = cr*sp*cy + sr*cp*sy
    qz = cr*cp*sy - sr*cp*cy
    return [qw, qx, qy, qz]

def quat_to_R(qw, qx, qy, qz):
    n = math.sqrt(qw*qw + qx*qx + qy*qy + qz*qz)
    qw, qx, qy, qz = qw/n, qx/n, qy/n, qz/n
    R = np.array([
        [1-2*(qy*qy+qz*qz),   2*(qx*qy-qz*qw),   2*(qx*qz+qy*qw)],
        [  2*(qx*qy+qz*qw), 1-2*(qx*qx+qz*qz),   2*(qy*qz-qx*qw)],
        [  2*(qx*qz-qy*qw),   2*(qy*qz+qx*qw), 1-2*(qx*qx+qy*qy)]
    ], dtype=float)
    return R

# ---------------- 自适应外环控制律 ----------------
def adaptive_controller(pos, vel, att, ang_vel, posd, attd, dhat, jifen, dt, t, fhat=None):
    """fhat: 世界系残差力 [fx,fy,fz]，若为 None 则视为零"""
    fx, fy, fz = (0.0, 0.0, 0.0) if (fhat is None) else (float(fhat[0]), float(fhat[1]), float(fhat[2]))
    x,y,z = pos[0][-1], pos[1][-1], pos[2][-1]
    u,v,w = vel[0][-1], vel[1][-1], vel[2][-1]
    phi,theta,psi = att[0][-1], att[1][-1], att[2][-1]
    xd,yd,zd,psid = posd[0][-1],posd[1][-1],posd[2][-1],attd[2][-1]
    dx_hat,dy_hat,dz_hat,dphi_hat,dtheta_hat,dpsi_hat = dhat
    xphi,xtheta,xpsi = jifen

    # 期望速度/加速度
    xd_dot = (posd[0][-1]-posd[0][-2])/dt if len(posd[0])>=2 else 0.0
    yd_dot = (posd[1][-1]-posd[1][-2])/dt if len(posd[1])>=2 else 0.0
    zd_dot = (posd[2][-1]-posd[2][-2])/dt if len(posd[2])>=2 else 0.0
    xd_dot2 = (((posd[0][-1]-posd[0][-2])/dt - (posd[0][-2]-posd[0][-3])/dt)/dt) if len(posd[0])>=3 else 0.0
    yd_dot2 = (((posd[1][-1]-posd[1][-2])/dt - (posd[1][-2]-posd[1][-3])/dt)/dt) if len(posd[1])>=3 else 0.0
    zd_dot2 = (((posd[2][-1]-posd[2][-2])/dt - (posd[2][-2]-posd[2][-3])/dt)/dt) if len(posd[2])>=3 else 0.0

    # 误差
    ex = x - xd; ey = y - yd; ez = z - zd
    eu = u - xd_dot + cx*ex
    ev = v - yd_dot + cy*ey
    ew = w - zd_dot + cz*ez

    # “去残差”后的误差（把 fhat/m 当作已知扰动）
    eu_tilde = eu - fx / UAV_mass
    ev_tilde = ev - fy / UAV_mass
    ew_tilde = ew - fz / UAV_mass

    # 误差微分
    ex_dot = eu - cx*ex
    ey_dot = ev - cy*ey
    ez_dot = ew - cz*ez

    # 期望加速度（保持只减 \hat d，不再在这里直接加 fhat）
    u_dot = -cu*eu - ex + xd_dot2 - cx*ex_dot
    v_dot = -cv*ev - ey + yd_dot2 - cy*ey_dot
    w_dot = -cw*ew - ez + zd_dot2 - cz*ez_dot

    # 自适应律：用 *_tilde 更新 \hat d，避免与 fhat 打架
    dz_hat += lamz*ew_tilde*dt
    dx_hat += lamx*eu_tilde*dt
    dy_hat += lamy*ev_tilde*dt

    # 油门
    thrust_force = -(w_dot - dz_hat - G) * UAV_mass / (math.cos(phi)*math.cos(theta))
    throttle = clip((thrust_force/(UAV_mass*G))*0.5 + 0.5, 0.0, 1.0)

    # 侧向加速度只扣 \hat d（fhat 已进入 *_tilde）
    accel_x_desired = clip(u_dot - dx_hat,  -5.0, 5.0)
    accel_y_desired = clip(-(v_dot - dy_hat), -5.0, 5.0)

    # 姿态指令
    roll_desired  = -(accel_y_desired*math.cos(psi) - accel_x_desired*math.sin(psi))/G
    pitch_desired =  (accel_x_desired*math.cos(psi) + accel_y_desired*math.sin(psi))/G
    max_angle = math.radians(30)
    roll_desired  = clip(roll_desired,  -max_angle, max_angle)
    pitch_desired = clip(pitch_desired, -max_angle, max_angle)
    yaw_desired = psid

    dhat_new = [dx_hat, dy_hat, dz_hat, dphi_hat, dtheta_hat, dpsi_hat]
    jifen_new = [xphi, xtheta, xpsi]
    return throttle, roll_desired, pitch_desired, yaw_desired, dhat_new, jifen_new, (u_dot, v_dot, w_dot)

# ---------------- Meta-PINN 包装（门控/限幅/率限制 + 稳定在线学习 + 可选 cond） ----------------
class MetaPINNWrapper:
    def __init__(self, feature_keys, scaler_path=None, load_path=None,
                 lr=1e-3, buffer_size=4096, batch_size=256, update_every=30):
        self.feature_keys = list(feature_keys)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # scaler
        self.scaler = None
        self.in_dim_from_scaler = None
        if scaler_path and os.path.isfile(scaler_path):
            npz = np.load(scaler_path)
            self.scaler = {"mean": npz["x_mean"].astype(np.float32),
                           "std":  npz["x_std"].astype(np.float32)}
            self.in_dim_from_scaler = int(self.scaler["mean"].shape[0])
            print(f"[Meta-PINN] Loaded input scaler: {scaler_path} | dim={self.in_dim_from_scaler}")

        # 读取权重
        self._sd = None
        if load_path and os.path.isfile(load_path):
            sd = torch.load(load_path, map_location="cpu")
            self._sd = sd.get("model_state_dict", sd)
            print(f"[Meta-PINN] Loaded weights: {load_path}")

        # 运行态
        self.model = None
        self.opt = None
        self.lr = lr
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.update_every = update_every
        self.step = 0
        self.task_id = 0

        # 门控运行量
        self.ema_err = 0.0
        self.last_fhat = np.zeros(3, dtype=np.float32)

    def _ensure_model(self, in_dim_runtime: int):
        if self.model is not None: return
        if self.in_dim_from_scaler is not None:
            assert in_dim_runtime == self.in_dim_from_scaler, \
                f"在线特征维度({in_dim_runtime})与 x_scaler({self.in_dim_from_scaler})不一致"

        # 从 state_dict 推断结构
        if self._sd is not None:
            num_tasks, task_dim = map(int, self._sd["task_embeddings"].shape)
            hidden_dim = int(self._sd["fc1.weight"].shape[0])
            input_dim_from_sd = int(self._sd["fc1.weight"].shape[1] - task_dim)
            assert input_dim_from_sd == in_dim_runtime, \
                f"state_dict 输入维度({input_dim_from_sd})与在线特征({in_dim_runtime})不一致"
            use_cond_mod = ("cond2beta.0.weight" in self._sd)
            cond_dim = int(self._sd["cond2beta.0.weight"].shape[1]) if use_cond_mod else 1
            use_uncertainty = ("log_vars" in self._sd)
        else:
            num_tasks, task_dim, hidden_dim = 1, 128, 384
            use_cond_mod, cond_dim, use_uncertainty = True, 1, True

        self.model = MetaPINN(
            input_dim=in_dim_runtime, num_tasks=num_tasks,
            task_dim=task_dim, hidden_dim=hidden_dim,
            use_uncertainty=use_uncertainty,
            cond_dim=cond_dim, use_cond_mod=use_cond_mod, cond_mod_from='target',
            beta_min=0.15, beta_max=8.0
        ).to(self.device)

        if self._sd is not None:
            self.model.load_state_dict(self._sd, strict=False)

        # 冻结除 task 嵌入/β 以外的参数
        for p in self.model.parameters(): p.requires_grad = False
        self.model.task_embeddings.requires_grad = True
        if hasattr(self.model, "task_beta_logscale"):
            self.model.task_beta_logscale.requires_grad = True

        self.opt = torch.optim.SGD(
            [p for p in [self.model.task_embeddings, getattr(self.model, "task_beta_logscale", None)] if p is not None],
            lr=self.lr, momentum=0.0
        )

    def _norm(self, x):
        if self.scaler is None: return x
        m, s = self.scaler["mean"], self.scaler["std"]
        s = np.where(s < 1e-6, 1.0, s)
        return (x - m) / s

    def _gate(self, x_np: np.ndarray):
        # 简单 OOD 门控 + 不确定性门控（用在线EMA误差替代）
        if self.scaler is None: g_ood = 1.0
        else:
            z = (x_np - self.scaler['mean']) / np.maximum(self.scaler['std'], 1e-6)
            d2 = float(np.mean(z*z))
            g_ood = math.exp(-d2 / 12.0)  # tau=8 可调
        g_unc = 1.0 / (1.0 + self.ema_err / 0.3)  # 以 0.3g 归一（单位化后）
        g = max(0.0, min(1.0, g_ood * g_unc))
        return g

    def predict(self, feat_np: np.ndarray, cond_val: float = None) -> np.ndarray:
        feat_np = feat_np.astype(np.float32)
        self._ensure_model(in_dim_runtime=feat_np.shape[0])
        x = torch.tensor(self._norm(feat_np), dtype=torch.float32, device=self.device).unsqueeze(0)

        # 可选：条件调制（若模型提供该接口）
        if (cond_val is not None) and hasattr(self.model, "_beta_from_c"):
            c = torch.tensor([[cond_val]], dtype=torch.float32, device=self.device)
            with torch.no_grad():
                _ = self.model._beta_from_c(c, task_id=self.task_id)

        self.model.eval()
        with torch.no_grad():
            fhat = self.model._pred_physical(x, task_id=self.task_id).squeeze(0).cpu().numpy()

        # 门控 + 限幅 + 率限制
        g = self._gate(feat_np)
        mag_limit = 0.6 * 9.81 * UAV_mass          # <= 0.5g
        fhat = np.clip(fhat, -mag_limit, mag_limit)
        step_lim = 0.15 * 9.81 * UAV_mass           # 每步 <= 0.2g
        delta = np.clip(fhat - self.last_fhat, -step_lim, step_lim)
        fhat_used = self.last_fhat + g * delta
        self.last_fhat = fhat_used.astype(np.float32)
        return self.last_fhat

    def push(self, feat_np: np.ndarray, target_np: np.ndarray, cur_vel=None, att=None, thr=None):
        # —— 样本筛选：低速/大姿态/油门贴边 不学习 ——
        if cur_vel is not None and np.linalg.norm(cur_vel) < 0.3:
            return
        if att is not None and (abs(att[0]) > math.radians(30) or abs(att[1]) > math.radians(30)):
            return
        if thr is not None and (thr < 0.03 or thr > 0.97):
            return

        # 维护无量纲误差的 EMA（以 g 为尺度）
        err = float(np.linalg.norm(target_np - self.last_fhat) / max(1e-6, 9.81*UAV_mass))
        self.ema_err = 0.9*self.ema_err + 0.1*err
        self.buffer.append((feat_np.astype(np.float32), target_np.astype(np.float32)))


    def online_update(self):
        if (self.model is None) or len(self.buffer) < max(64, self.batch_size//2): return
        idx = np.random.choice(len(self.buffer), size=min(self.batch_size, len(self.buffer)), replace=False)
        xs = np.stack([self.buffer[i][0] for i in idx], axis=0)
        ys = np.stack([self.buffer[i][1] for i in idx], axis=0)
        x = torch.tensor(self._norm(xs), dtype=torch.float32, device=self.device)
        y = torch.tensor(ys, dtype=torch.float32, device=self.device)

        self.model.train()
        self.opt.zero_grad()
        f_pred = self.model._pred_physical(x, task_id=self.task_id)
        loss = F.smooth_l1_loss(f_pred, y, beta=5.0)  # Huber
        loss.backward()

        # 只让当前 task 的 embedding / β 更新
        with torch.no_grad():
            if self.model.task_embeddings.grad is not None:
                mask = torch.zeros_like(self.model.task_embeddings); mask[self.task_id] = 1.0
                self.model.task_embeddings.grad *= mask
            if hasattr(self.model, "task_beta_logscale") and (self.model.task_beta_logscale.grad is not None):
                maskb = torch.zeros_like(self.model.task_beta_logscale); maskb[self.task_id] = 1.0
                self.model.task_beta_logscale.grad *= maskb

        self.opt.step()

    def maybe_update(self):
        self.step += 1
        if self.step % self.update_every == 0:
            self.online_update()

# ---------------- 控制器（集成：残差→自适应） ----------------
class SimpleFlightController:
    def __init__(self, meta_wrapper: MetaPINNWrapper, feature_keys, use_meta=True, online_learn=True):
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)

        self.dt = 0.01
        self.simulation_time = 0.0
        self.hover_throttle = HOVER_THROTTLE

        self.prev_vel = None
        self.prev_posd = None

        self.dhat  = [0.0]*6
        self.jifen = [0.0]*3

        self.pos_history = [[],[],[]]
        self.vel_history = [[],[],[]]
        self.att_history = [[],[],[]]
        self.ang_vel_history = [[],[],[]]
        self.posd_history = [[],[],[]]
        self.attd_history = [[],[],[]]

        self.meta = meta_wrapper
        self.use_meta = use_meta
        self.online_learn = online_learn
        self.feature_keys = list(feature_keys)

        self.data_log = {k: [] for k in ["t","p","p_d","v","v_d","q","R","w","att","att_d","cmd","T_sp","q_sp","hover_throttle","fa","pwm"]}

        print(f"SimpleFlightController initialized | features={self.feature_keys}")

    def apply_wind_profile(self, profile, t):
        if profile["kind"] == "const":
            mag = profile["mag"]
        elif profile["kind"] == "sin":
            mag = profile["mag_mean"] + profile["mag_amp"]*np.sin(2*np.pi*profile["freq_hz"]*t)
        elif profile["kind"] == "gust":
            mag = profile["mag"] + np.random.normal(0, profile.get("noise_std",1.0))
        else:
            mag = 0.0
        X = profile["dir"][0] * mag
        Y = profile["dir"][1] * mag
        Z = profile["dir"][2] * mag
        self.client.simSetWind(airsim.Vector3r(X, Y, Z))

    def get_state(self):
        st = self.client.getMultirotorState()
        p = st.kinematics_estimated.position
        v = st.kinematics_estimated.linear_velocity
        q = st.kinematics_estimated.orientation
        w = st.kinematics_estimated.angular_velocity
        qw,qx,qy,qz = q.w_val, q.x_val, q.y_val, q.z_val
        roll,pitch,yaw = quaternion_to_euler(qx,qy,qz,qw)
        R = quat_to_R(qw,qx,qy,qz)
        return (
            [p.x_val, p.y_val, p.z_val],
            [v.x_val, v.y_val, v.z_val],
            [roll, pitch, yaw],
            [w.x_val, w.y_val, w.z_val],
            [qw,qx,qy,qz],
            R
        )

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
                    buf[i] = buf[i][-max_history:]

    def _build_features(self, cur_pos, cur_vel, cur_att, cur_ang, cur_quat, cur_R,
                        des_pos, des_att, q_sp, thr, v_d):
        parts = []
        for key in self.feature_keys:
            if   key == "p":     parts.append(np.asarray(cur_pos, dtype=np.float32))
            elif key == "v":     parts.append(np.asarray(cur_vel, dtype=np.float32))
            elif key == "q":     parts.append(np.asarray(cur_quat, dtype=np.float32))
            elif key == "w":     parts.append(np.asarray(cur_ang, dtype=np.float32))
            elif key == "R":     parts.append(cur_R.astype(np.float32).reshape(-1))
            elif key == "p_d":   parts.append(np.asarray(des_pos, dtype=np.float32))
            elif key == "att":   parts.append(np.asarray(cur_att, dtype=np.float32))
            elif key == "att_d": parts.append(np.asarray(des_att, dtype=np.float32))
            elif key == "v_d":   parts.append(np.asarray(v_d, dtype=np.float32))
            elif key == "q_sp":  parts.append(np.asarray(q_sp, dtype=np.float32))
            elif key == "T_sp":  parts.append(np.asarray([thr], dtype=np.float32))
            elif key == "pwm":
                try:
                    rs = self.client.getRotorStates()
                    spds = [rs.rotors[i]['speed'] for i in range(4)]
                except Exception:
                    spds = [0.0,0.0,0.0,0.0]
                parts.append(np.asarray(spds, dtype=np.float32))
            else:
                parts.append(np.zeros(1, dtype=np.float32))
        return np.concatenate(parts, axis=0)

    def send_control_to_airsim(self, throttle, roll_desired, pitch_desired, yaw_desired):
        self.client.moveByRollPitchYawThrottleAsync(
            roll_desired, pitch_desired, yaw_desired, throttle, duration=self.dt
        )

    def _log_for_fig(self, t, cur_pos, des_pos, cur_vel, v_d, cur_quat, cur_R, cur_ang_vel, cur_att,
                     des_att, thr, r_des, p_des, y_des, q_sp=None, fa_obs=None):
        self.data_log["t"].append(t)
        self.data_log["p"].append(cur_pos[:])
        self.data_log["p_d"].append(des_pos[:])
        self.data_log["v"].append(cur_vel[:])
        self.data_log["v_d"].append(v_d[:])
        self.data_log["q"].append(cur_quat[:])
        self.data_log["R"].append(cur_R.tolist())
        self.data_log["w"].append(cur_ang_vel[:])
        cur_att_plot = [cur_att[0], -cur_att[1], cur_att[2]] 
        self.data_log["att"].append(cur_att_plot)
        self.data_log["att_d"].append([r_des, p_des, y_des])
        self.data_log["cmd"].append([thr, r_des, p_des, y_des])
        
        # Additional data for CSV export (matching data_collection_all.py format)
        self.data_log["T_sp"].append([thr])
        self.data_log["q_sp"].append(q_sp if q_sp is not None else [1.0, 0.0, 0.0, 0.0])
        self.data_log["hover_throttle"].append([self.hover_throttle])
        self.data_log["fa"].append(fa_obs.tolist() if fa_obs is not None else [0.0, 0.0, 0.0])
        
        # Get rotor speeds
        try:
            rs = self.client.getRotorStates()
            pwm_speeds = [rs.rotors[i]['speed'] for i in range(4)]
        except Exception:
            pwm_speeds = [0.0, 0.0, 0.0, 0.0]
        self.data_log["pwm"].append(pwm_speeds)

    def _save_figures(self, out_dir: Path, base_name: str):
        out_dir.mkdir(parents=True, exist_ok=True)
        pos_arr = np.array(self.data_log["p"])
        des_arr = np.array(self.data_log["p_d"])
        att_arr = np.array(self.data_log["att"])
        attd_arr= np.array(self.data_log["att_d"])
        t_arr   = np.array(self.data_log["t"])
        cmd_arr = np.array(self.data_log["cmd"])

        fig = plt.figure(figsize=(12,8))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(pos_arr[:,0], pos_arr[:,1], pos_arr[:,2], 'b-', label='Actual', linewidth=2)
        ax.plot(des_arr[:,0], des_arr[:,1], des_arr[:,2], 'r--', label='Desired', linewidth=2)
        ax.scatter(pos_arr[0,0], pos_arr[0,1], pos_arr[0,2], c='g', s=80, marker='o', label='Start')
        ax.scatter(pos_arr[-1,0], pos_arr[-1,1], pos_arr[-1,2], c='r', s=80, marker='s', label='End')
        ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.set_zlabel('Z (m)')
        ax.set_title('3D Trajectory'); ax.legend(); ax.grid(True)
        plt.tight_layout(); fig.savefig(out_dir / f"{base_name}_traj3d.png"); plt.close(fig)

        fig, ax = plt.subplots(figsize=(8,6))
        ax.plot(pos_arr[:,0], pos_arr[:,1], 'b-', label='Actual', linewidth=2)
        ax.plot(des_arr[:,0], des_arr[:,1], 'r--', label='Desired', linewidth=2)
        ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.set_title('2D Trajectory (X-Y)')
        ax.legend(); ax.grid(True)
        plt.tight_layout(); fig.savefig(out_dir / f"{base_name}_traj2d.png"); plt.close(fig)

        fig, axes = plt.subplots(2,2, figsize=(12,8))
        labels = ['Throttle (0-1)','Roll cmd (deg)','Pitch cmd (deg)','Yaw cmd (deg)']
        series = [cmd_arr[:,0], np.degrees(cmd_arr[:,1]), np.degrees(cmd_arr[:,2]), np.degrees(cmd_arr[:,3])]
        for i in range(4):
            r,c = i//2, i%2
            axes[r,c].plot(t_arr, series[i], linewidth=2)
            axes[r,c].set_xlabel('Time (s)'); axes[r,c].set_ylabel(labels[i]); axes[r,c].grid(True)
        fig.suptitle('Control Commands vs Time')
        plt.tight_layout(); fig.savefig(out_dir / f"{base_name}_ctrl.png"); plt.close(fig)

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        for i, label in enumerate(['X', 'Y', 'Z']):
            axes[0, i].plot(t_arr, pos_arr[:, i], 'b-', label='Actual', linewidth=2)
            axes[0, i].plot(t_arr, des_arr[:, i], 'r--', label='Desired', linewidth=2)
            axes[0, i].set_xlabel('Time (s)')
            axes[0, i].set_ylabel(f'{label} (m)'); axes[0, i].set_title(f'{label}-Position Tracking')
            axes[0, i].legend(); axes[0, i].grid(True)
        for i, label in enumerate(['Roll', 'Pitch', 'Yaw']):
            axes[1, i].plot(t_arr, np.degrees(att_arr[:, i]), 'b-', label='Actual', linewidth=2)
            axes[1, i].plot(t_arr, np.degrees(attd_arr[:, i]), 'r--', label='Desired', linewidth=2)
            axes[1, i].set_xlabel('Time (s)')
            axes[1, i].set_ylabel(f'{label} (deg)'); axes[1, i].set_title(f'{label} Tracking')
            axes[1, i].legend(); axes[1, i].grid(True)
        plt.tight_layout()
        fig.savefig(out_dir / f"{base_name}_track.png"); plt.close(fig)
       
    def _save_csv(self, out_dir: Path, base_name: str):
        """Save flight data to CSV file in the same format as data_collection_all.py"""
        out_dir.mkdir(parents=True, exist_ok=True)
        out_csv = out_dir / f"{base_name}.csv"
        
        with open(out_csv, "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["idx","t","p","p_d","v","v_d","q","R","w","T_sp","q_sp","hover_throttle","fa","pwm"])
            for i in range(len(self.data_log["t"])):
                wr.writerow([
                    i,
                    f"{self.data_log['t'][i]:.14g}",
                    str(self.data_log['p'][i]),
                    str(self.data_log['p_d'][i]),
                    str(self.data_log['v'][i]),
                    str(self.data_log['v_d'][i]),
                    str(self.data_log['q'][i]),
                    str(self.data_log['R'][i]),
                    str(self.data_log['w'][i]),
                    str(self.data_log['T_sp'][i]),
                    str(self.data_log['q_sp'][i]),
                    str(self.data_log['hover_throttle'][i]),
                    str(self.data_log['fa'][i]),
                    str(self.data_log['pwm'][i]),
                ])
        print(f"[CSV保存] {out_csv}")

    @staticmethod
    def _wrap_angle_rad(e):
        # wrap到[-pi, pi]，避免359° vs 0°造成的巨大误差
        return (e + np.pi) % (2*np.pi) - np.pi

    def _print_avg_tracking_error(self, label: str):
        # 位置误差
        pos = np.array(self.data_log["p"], dtype=np.float32)       # [T,3]
        des = np.array(self.data_log["p_d"], dtype=np.float32)     # [T,3]
        e_pos = pos - des                                          # [T,3]
        mae_xyz  = np.mean(np.abs(e_pos), axis=0)                  # [3]
        mae_3d   = np.mean(np.linalg.norm(e_pos, axis=1))          # 标量
        rmse_3d  = np.sqrt(np.mean(np.sum(e_pos**2, axis=1)))      # 标量

        # 姿态误差（弧度），注意包角
        att  = np.array(self.data_log["att"], dtype=np.float32)    # [T,3] rad
        attd = np.array(self.data_log["att_d"], dtype=np.float32)  # [T,3] rad
        e_att = att - attd
        for k in range(3):
            e_att[:, k] = self._wrap_angle_rad(e_att[:, k])
        att_mae_deg = np.degrees(np.mean(np.abs(e_att), axis=0))   # [3] deg

        print(
            f"[Avg Tracking Error] {label} | "
            f"Pos MAE XYZ (m)=({mae_xyz[0]:.3f}, {mae_xyz[1]:.3f}, {mae_xyz[2]:.3f}), "
            f"3D-MAE={mae_3d:.3f} m, 3D-RMSE={rmse_3d:.3f} m | "
            f"Att MAE (deg)=({att_mae_deg[0]:.2f}, {att_mae_deg[1]:.2f}, {att_mae_deg[2]:.2f})"
        )


    def run_simulation(self, total_time=60.0, trajectory_func=None, profile=None,
                       trajectory_type="random", wind_tag=None):
        self.simulation_time = 0.0
        self.client.takeoffAsync().join()
        time.sleep(1.5)
        mass = UAV_mass

        if trajectory_func is None:
            trajectory_func = test3_random_spline_trajectory()

        while self.simulation_time < total_time:
            t0 = time.time()
            if profile is not None:
                self.apply_wind_profile(profile, self.simulation_time)

            (cur_pos, cur_vel, cur_att, cur_ang_vel, cur_quat, cur_R) = self.get_state()
            xd,yd,zd,psid = trajectory_func(self.simulation_time)
            des_pos = [xd,yd,zd]; des_att=[0.0,0.0,psid]

            if self.prev_posd is None:
                v_d = [0.0,0.0,0.0]
            else:
                v_d = [(des_pos[i]-self.prev_posd[i])/self.dt for i in range(3)]

            self.update_history(cur_pos, cur_vel, cur_att, cur_ang_vel, des_pos, des_att)

            if len(self.pos_history[0]) >= 3:
                # === 特征构造（与离线一致） ===
                # 先用“原始”姿态指令占位构造 q_sp（先不含 PINN 并联更改）
                r_tmp, p_tmp, y_tmp = 0.0, 0.0, des_att[2]
                q_sp = euler_to_quat(r_tmp, -p_tmp, y_tmp)
                feat = self._build_features(cur_pos, cur_vel, cur_att, cur_ang_vel, cur_quat, cur_R,
                                            des_pos, des_att, q_sp, 0.5, v_d)

                # === 预测残差（含 OOD 门控/限幅/率限制 & 可选 cond）===
                fhat = None
                if self.use_meta:
                    speed = float(np.linalg.norm(cur_vel))  # 作为 cond_val
                    fhat = self.meta.predict(feat, cond_val=speed)

                # 记录 dz_hat 以便做抗积分
                dz_prev = self.dhat[2]

                # === 自适应控制（把 fhat 喂给自适应误差，避免并联打架）===
                thr, r_des, p_des, y_des, self.dhat, self.jifen, (u_dot, v_dot, w_dot) = adaptive_controller(
                    self.pos_history, self.vel_history, self.att_history, self.ang_vel_history,
                    self.posd_history, self.attd_history, self.dhat, self.jifen, self.dt, self.simulation_time,
                    fhat=fhat
                )
                q_sp = euler_to_quat(r_des, -p_des, y_des)

                # >>>>>>> 新增：竖直通道小比例前馈 + 抗积分 <<<<<<<
                if self.use_meta and (fhat is not None):
                    g_ff = 0.3  # 0.2~0.4 之间调
                    b_z = cur_R[:, 2]
                    thrust_add = g_ff * float(np.dot(fhat, -b_z))           # 只用沿 -b_z 的分量
                    thrust_force = (thr / HOVER_THROTTLE) * mass * G + thrust_add
                    thr = clip((thrust_force / (mass * G)) * 0.5 + 0.5, 0.0, 1.0)

                thr_sat = (thr < 0.02) or (thr > 0.98)
                if thr_sat:
                    self.dhat[2] = dz_prev   # throttle 贴边时冻结 dz_hat，防止积分冲掉


                # === 下发控制 ===
                self.send_control_to_airsim(thr, r_des, p_des, y_des)

                # === 在线学习目标：fa_obs = m a - thrust_world - g ===
                fa_obs = None  # Initialize fa_obs
                if self.online_learn and self.prev_vel is not None:
                    #a_world = (np.array(cur_vel) - np.array(self.prev_vel))/self.dt
                    # --- 一阶低通（fc≈4Hz） ---
                    if not hasattr(self, "_a_lp"):
                        self._a_lp = np.array([0.0, 0.0, 0.0], dtype=np.float32)
                    a_raw = (np.array(cur_vel) - np.array(self.prev_vel)) / self.dt
                    alpha = 2 * math.pi * 4.0 * self.dt         # 简单一阶（可再按需调大/调小）
                    self._a_lp = (1 - alpha) * self._a_lp + alpha * a_raw
                    a_world = self._a_lp

                    b_z = cur_R[:,2]
                    thrust_world = -(thr/HOVER_THROTTLE) * mass * G * b_z
                    fa_obs = mass*a_world - thrust_world - np.array([0.0,0.0,mass*G], dtype=np.float32)
                    self.meta.push(feat, fa_obs.astype(np.float32), cur_vel=cur_vel, att=cur_att, thr=thr)
                    self.meta.maybe_update()

                # === 仅为出图记录 ===
                self._log_for_fig(self.simulation_time, cur_pos, des_pos, cur_vel, v_d, cur_quat, cur_R,
                                  cur_ang_vel, cur_att, [r_des, p_des, y_des], thr, r_des, p_des, y_des, q_sp, fa_obs)

            self.prev_vel = cur_vel[:]
            self.prev_posd = des_pos[:]
            self.simulation_time += self.dt
            time.sleep(max(0.0, self.dt - (time.time()-t0)))

        # 落地 
        self.client.landAsync().join()
        self.client.armDisarm(False)
        self.client.enableApiControl(False)

        vehicle = "simpleflight"
        condition = wind_tag if wind_tag else "nowind"
        base = f"{vehicle}_{trajectory_type}_metapinn_{condition}"
        self._save_figures(RESULTS_DIR, base)
        self._save_csv(RESULTS_DIR, base)
        print(f"[保存图片] {RESULTS_DIR / (base+'_traj3d.png')}")
        self._print_avg_tracking_error(label=f"{trajectory_type} | {condition}")



# ---------------- 轨迹 ----------------
def test2(t):
    x = 10.0*math.sin(0.5*t)
    y = 10.0*math.sin(0.5*t)*math.cos(0.5*t)
    z = -10.0 - 2.0*t
    yaw = 0.0
    return x,y,z,yaw

def test3_random_spline_trajectory():
    import numpy as np
    from scipy.interpolate import CubicSpline
    total_time = 60.0
    num_points = 15
    t_points = np.linspace(0, total_time, num_points)
    x_points = np.random.uniform(-15, 15, num_points)
    y_points = np.random.uniform(-15, 15, num_points)
    z_points = np.random.uniform(-15, -5, num_points)
    yaw_points = np.random.uniform(-np.pi, np.pi, num_points)
    x_s, y_s, z_s, yaw_s = CubicSpline(t_points, x_points), CubicSpline(t_points, y_points), CubicSpline(t_points, z_points), CubicSpline(t_points, yaw_points)
    def f(t):
        t = np.clip(t, 0, total_time)
        return float(x_s(t)), float(y_s(t)), float(z_s(t)), 0.0
    return f

# ---------------- 复位 ----------------
def reset_to_home():
    c = airsim.MultirotorClient(); c.confirmConnection()
    try: c.simSetWind(airsim.Vector3r(0.0, 0.0, 0.0))
    except Exception: pass
    c.reset(); time.sleep(1.5)

# ---------------- 入口 ----------------
def main():
    ap = argparse.ArgumentParser(description="AirSim adaptive + Meta-PINN (save figs only, robust online)")
    ap.add_argument('--mode', choices=['train','test'], default='test')
    ap.add_argument('--idx', type=int, default=0)
    ap.add_argument('--mp_load', type=str, default='saved_models/meta_pinn_offline/meta_pinn_last.pth',
                    help='离线 Meta-PINN 权重路径')
    ap.add_argument('--mp_scaler', type=str, default='saved_models/meta_pinn_offline/x_scaler.npz',
                    help='输入标准化 npz（x_mean/x_std）')
    ap.add_argument('--features', type=str, default='v,q,pwm',
                    help='与离线训练一致的特征列表，逗号分隔、顺序一致')
    ap.add_argument('--no_meta', action='store_true', help='禁用 Meta-PINN 预测')
    ap.add_argument('--no_online', action='store_true', help='禁用在线学习')
    args = ap.parse_args()

    feature_keys = [s.strip() for s in args.features.split(',') if s.strip()]

    all_items = list(PROFILES_ALL.values())
    train_list = [(p, WIND_CONDITIONS[p["tag"]]) for p in all_items if WIND_CONDITIONS.get(p["tag"]) in TRAIN_CONDS]
    test_list  = [(p, WIND_CONDITIONS[p["tag"]]) for p in all_items if WIND_CONDITIONS.get(p["tag"]) in TEST_CONDS]

    def run_one(profile, condition, traj_fn, traj_type, sim_time):
        print(f"[Run] {traj_type} | {condition} | profile={profile['tag']} | {sim_time}s")
        reset_to_home()
        meta = MetaPINNWrapper(feature_keys=feature_keys,
                               scaler_path=args.mp_scaler,
                               load_path=args.mp_load,
                               lr=1e-3, update_every=30)
        ctl = SimpleFlightController(meta_wrapper=meta,
                                     feature_keys=feature_keys,
                                     use_meta=not args.no_meta,
                                     online_learn=not args.no_online)
        ctl.run_simulation(total_time=sim_time,
                           trajectory_func=traj_fn,
                           profile=profile,
                           trajectory_type=traj_type,
                           wind_tag=condition)

    if args.mode and args.idx is not None:
        if args.mode == 'train':
            items = train_list
            print("可选训练风况：")
            for i,(p,c) in enumerate(items): print(f"[{i}] {p['tag']} -> {c}")
            i = max(0, min(args.idx, len(items)-1))
            run_one(items[i][0], items[i][1], test3_random_spline_trajectory(), "random", 60.0)
        else:
            items = test_list
            print("可选测试风况：")
            for i,(p,c) in enumerate(items): print(f"[{i}] {p['tag']} -> {c}")
            i = max(0, min(args.idx, len(items)-1))
            run_one(items[i][0], items[i][1], test2, "fig8", 30.0)
        return

    print("未指定参数，默认批量运行全部训练与测试风况（只在 result/ 产出图片）")
    # for p,c in train_list:
    #     run_one(p, c, test3_random_spline_trajectory(), "random", 60.0)

    for p,c in test_list:
        run_one(p, c, test2, "fig8", 30.0)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[中断] 用户手动停止")
    except Exception as e:
        import traceback
        print(f"[错误] {e}")
        traceback.print_exc()
