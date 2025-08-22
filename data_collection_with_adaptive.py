#!/usr/bin/env python
# -*- coding: utf-8 -*-
import airsim
import time, math, csv
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import torch, torch.nn as nn
from pathlib import Path
from config import *

HOVER_THROTTLE = 0.594

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

# 输出目录
OUT_DIR_TRAIN = Path("logs_random_profiles")
OUT_DIR_TEST  = Path("logs_test_fig8")
# OUT_DIR_TRAIN = Path("train_wind")
# OUT_DIR_TEST = Path("test_wind")

# =============== 姿态/旋转工具 ===============
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
    qz = cr*cp*sy - sr*sp*cy
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

# =============== 控制器（自适应律） ===============
class AdaptiveNeuralNetwork(nn.Module):
    def __init__(self, input_dim=12, hidden_dim=64, output_dim=3):
        super().__init__()
        self.hidden_layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU()
        )
        self.adaptive_layer = nn.Linear(hidden_dim, output_dim)
        for p in self.hidden_layers.parameters():
            p.requires_grad = False
        nn.init.xavier_uniform_(self.adaptive_layer.weight, gain=0.1)
        nn.init.zeros_(self.adaptive_layer.bias)
    def forward(self, x):
        return self.adaptive_layer(self.hidden_layers(x))

def nn_adaptive_controller(pos, vel, att, ang_vel, posd, attd, dhat, jifen, dt, t):
    x,y,z = pos[0][-1], pos[1][-1], pos[2][-1]
    u,v,w = vel[0][-1], vel[1][-1], vel[2][-1]
    phi,theta,psi = att[0][-1], att[1][-1], att[2][-1]
    xd,yd,zd,psid = posd[0][-1],posd[1][-1],posd[2][-1],attd[2][-1]
    dx_hat,dy_hat,dz_hat,dphi_hat,dtheta_hat,dpsi_hat = dhat
    xphi,xtheta,xpsi = jifen
    g = 9.81

    xd_dot = (posd[0][-1]-posd[0][-2])/dt if len(posd[0])>=2 else 0.0
    yd_dot = (posd[1][-1]-posd[1][-2])/dt if len(posd[1])>=2 else 0.0
    zd_dot = (posd[2][-1]-posd[2][-2])/dt if len(posd[2])>=2 else 0.0

    xd_dot2 = (((posd[0][-1]-posd[0][-2])/dt - (posd[0][-2]-posd[0][-3])/dt)/dt) if len(posd[0])>=3 else 0.0
    yd_dot2 = (((posd[1][-1]-posd[1][-2])/dt - (posd[1][-2]-posd[1][-3])/dt)/dt) if len(posd[1])>=3 else 0.0
    zd_dot2 = (((posd[2][-1]-posd[2][-2])/dt - (posd[2][-2]-posd[2][-3])/dt)/dt) if len(posd[2])>=3 else 0.0

    ez = z - zd; ew = w - zd_dot + cz*ez; ez_dot = ew - cz*ez
    w_dot = -cw*ew - ez + zd_dot2 - cz*ez_dot

    ex = x - xd; eu = u - xd_dot + cx*ex; ex_dot = eu - cx*ex
    u_dot = -cu*eu - ex + xd_dot2 - cx*ex_dot

    ey = y - yd; ev = v - yd_dot + cy*ey; ey_dot = ev - cy*ey
    v_dot = -cv*ev - ey + yd_dot2 - cy*ey_dot

    dz_hat += lamz*ew*dt
    dx_hat += lamx*eu*dt
    dy_hat += lamy*ev*dt

    thrust_force = -(w_dot - dz_hat - g) * UAV_mass / (math.cos(phi)*math.cos(theta))
    throttle = max(0.0, min(1.0, (thrust_force/(UAV_mass*9.81))*0.5 + 0.5))

    accel_x_desired = max(-5.0, min( 5.0, u_dot - dx_hat))
    accel_y_desired = max(-5.0, min( 5.0, -(v_dot - dy_hat)))

    roll_desired  = -(accel_y_desired*math.cos(psi) - accel_x_desired*math.sin(psi))/9.81
    pitch_desired =  (accel_x_desired*math.cos(psi) + accel_y_desired*math.sin(psi))/9.81
    max_angle = math.radians(30)
    roll_desired  = max(-max_angle, min(max_angle, roll_desired))
    pitch_desired = max(-max_angle, min(max_angle, pitch_desired))
    yaw_desired = psid

    dhat_new = [dx_hat, dy_hat, dz_hat, dphi_hat, dtheta_hat, dpsi_hat]
    jifen_new = [xphi, xtheta, xpsi]
    return throttle, roll_desired, pitch_desired, yaw_desired, dhat_new, jifen_new

# =============== 控制器封装 ===============
class SimpleFlightController:
    def __init__(self):
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

        print("SimpleFlightController initialized")

    def apply_wind_profile(self, profile, t):
        # 直接通过向量设置风场
        if profile["kind"] == "const":
            mag = profile["mag"]
        elif profile["kind"] == "sin":
            mag = profile["mag_mean"] + profile["mag_amp"]*np.sin(2*np.pi*profile["freq_hz"]*t)
        elif profile["kind"] == "gust":
            mag = profile["mag"] + np.random.normal(0, profile.get("noise_std",1.0))
        else:
            mag = 0.0
        # 这里假设风向是世界坐标系下的 (X, Y, Z)
        X = profile["dir"][0] * mag
        Y = profile["dir"][1] * mag
        Z = profile["dir"][2] * mag
        wind = airsim.Vector3r(X, Y, Z)
        self.client.simSetWind(wind)
        #print(f"[风场设置] t={t:.2f}s | tag={profile['tag']} | wind=({X:.2f}, {Y:.2f}, {Z:.2f})")
  
    def get_state(self):
        state = self.client.getMultirotorState()
        pos = state.kinematics_estimated.position
        position = [pos.x_val, pos.y_val, pos.z_val]
        vel = state.kinematics_estimated.linear_velocity
        velocity = [vel.x_val, vel.y_val, vel.z_val]
        ori = state.kinematics_estimated.orientation
        qw,qx,qy,qz = ori.w_val, ori.x_val, ori.y_val, ori.z_val
        roll,pitch,yaw = quaternion_to_euler(qx,qy,qz,qw)
        ang = state.kinematics_estimated.angular_velocity
        angular_velocity = [ang.x_val, ang.y_val, ang.z_val]
        R = quat_to_R(qw,qx,qy,qz)
        return position, velocity, [roll,pitch,yaw], angular_velocity, [qw,qx,qy,qz], R

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

    def send_control_to_airsim(self, throttle, roll_desired, pitch_desired, yaw_desired):
        self.client.moveByRollPitchYawThrottleAsync(
            roll_desired, pitch_desired, yaw_desired, throttle, duration=self.dt
        )

    # ===== 保存图像（含位置/姿态实际vs期望对比） =====
    def _save_figures(self, out_dir, base_name, data_log):
        pos_arr = np.array(data_log["p"])          # N x 3
        des_arr = np.array(data_log["p_d"])        # N x 3
        att_arr = np.array(data_log["att"])        # N x 3 (rad)
        attd_arr= np.array(data_log["att_d"])      # N x 3 (rad)
        t_arr   = np.array(data_log["t"])          # N
        cmd_arr = np.array(data_log["cmd"])        # N x 4  [thr,roll,pitch,yaw]

        # 3D 轨迹
        fig = plt.figure(figsize=(12,8))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(pos_arr[:,0], pos_arr[:,1], pos_arr[:,2], 'b-', label='Actual', linewidth=2)
        ax.plot(des_arr[:,0], des_arr[:,1], des_arr[:,2], 'r--', label='Desired', linewidth=2)
        ax.scatter(pos_arr[0,0], pos_arr[0,1], pos_arr[0,2], c='g', s=80, marker='o', label='Start')
        ax.scatter(pos_arr[-1,0], pos_arr[-1,1], pos_arr[-1,2], c='r', s=80, marker='s', label='End')
        ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.set_zlabel('Z (m)')
        ax.set_title('3D Trajectory'); ax.legend(); ax.grid(True)
        plt.tight_layout(); fig.savefig(out_dir / f"{base_name}_traj3d.png"); plt.close(fig)

        # 2D XY 轨迹
        fig, ax = plt.subplots(figsize=(8,6))
        ax.plot(pos_arr[:,0], pos_arr[:,1], 'b-', label='Actual', linewidth=2)
        ax.plot(des_arr[:,0], des_arr[:,1], 'r--', label='Desired', linewidth=2)
        ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.set_title('2D Trajectory (X-Y)')
        ax.legend(); ax.grid(True)
        plt.tight_layout(); fig.savefig(out_dir / f"{base_name}_traj2d.png"); plt.close(fig)

        # 控制输入时序（油门+姿态指令）
        fig, axes = plt.subplots(2,2, figsize=(12,8))
        labels = ['Throttle (0-1)','Roll cmd (deg)','Pitch cmd (deg)','Yaw cmd (deg)']
        series = [cmd_arr[:,0], np.degrees(cmd_arr[:,1]), np.degrees(cmd_arr[:,2]), np.degrees(cmd_arr[:,3])]
        for i in range(4):
            r,c = i//2, i%2
            axes[r,c].plot(t_arr, series[i], linewidth=2)
            axes[r,c].set_xlabel('Time (s)'); axes[r,c].set_ylabel(labels[i]); axes[r,c].grid(True)
        fig.suptitle('Control Commands vs Time')
        plt.tight_layout(); fig.savefig(out_dir / f"{base_name}_ctrl.png"); plt.close(fig)

        # ===== 位置/姿态：实际 vs 期望 =====
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        # 位置
        for i, label in enumerate(['X', 'Y', 'Z']):
            axes[0, i].plot(t_arr, pos_arr[:, i], 'b-', label='Actual', linewidth=2)
            axes[0, i].plot(t_arr, des_arr[:, i], 'r--', label='Desired', linewidth=2)
            axes[0, i].set_xlabel('Time (s)')
            axes[0, i].set_ylabel(f'{label} (m)')
            axes[0, i].set_title(f'{label}-Position Tracking')
            axes[0, i].legend(); axes[0, i].grid(True)
        # 姿态
        for i, label in enumerate(['Roll', 'Pitch', 'Yaw']):
            axes[1, i].plot(t_arr, np.degrees(att_arr[:, i]), 'b-', label='Actual', linewidth=2)
            axes[1, i].plot(t_arr, np.degrees(attd_arr[:, i]), 'r--', label='Desired', linewidth=2)
            axes[1, i].set_xlabel('Time (s)')
            axes[1, i].set_ylabel(f'{label} (deg)')
            axes[1, i].set_title(f'{label} Tracking')
            axes[1, i].legend(); axes[1, i].grid(True)
        plt.tight_layout()
        fig.savefig(out_dir / f"{base_name}_track.png"); plt.close(fig)

    def run_simulation(self, total_time=60.0, trajectory_func=None, profile=None,
                       trajectory_type="random", method="adaptive", wind_tag=None, out_root=None):
        # 运行日志（新增：att/att_d/cmd）
        data_log = {k: [] for k in ["t","p","p_d","v","v_d","q","R","w","att","att_d",
                                    "T_sp","q_sp","hover_throttle","fa","pwm","cmd"]}

        # 起飞
        self.simulation_time = 0.0
        self.client.takeoffAsync().join()
        time.sleep(2)
        g = 9.81; mass = UAV_mass

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
                thr, r_des, p_des, y_des, self.dhat, self.jifen = nn_adaptive_controller(
                    self.pos_history, self.vel_history, self.att_history, self.ang_vel_history,
                    self.posd_history, self.attd_history, self.dhat, self.jifen, self.dt, self.simulation_time
                )
                q_sp = euler_to_quat(r_des, -p_des, y_des)
                des_att = [r_des, -p_des, y_des]
                self.send_control_to_airsim(thr, r_des, p_des, y_des)

                a_world = np.zeros(3) if self.prev_vel is None else (np.array(cur_vel)-np.array(self.prev_vel))/self.dt
                b_z = cur_R[:,2]
                thrust_force = (thr/self.hover_throttle)*mass*g
                # thrust_force = (thr - 0.5) * 2 * mass * g
                # sum(c*w^2)
                f_thrust_world = -thrust_force*b_z
                fa = mass*a_world - f_thrust_world - np.array([0.0,0.0,mass*g])

                try:
                    rs = []
                    rotor_states = self.client.getRotorStates()
                    for i in range(4):
                        rs.append(rotor_states.rotors[i]['speed'])
                except Exception:
                    rs = [0.0,0.0,0.0,0.0]

                data_log["t"].append(self.simulation_time)
                data_log["p"].append(cur_pos[:])
                data_log["p_d"].append(des_pos[:])
                data_log["v"].append(cur_vel[:])
                data_log["v_d"].append(v_d[:])
                data_log["q"].append(cur_quat[:])
                data_log["R"].append(cur_R.tolist())
                data_log["w"].append(cur_ang_vel[:])
                data_log["att"].append(cur_att[:])
                data_log["att_d"].append(des_att[:])
                data_log["T_sp"].append([thr])
                data_log["q_sp"].append(q_sp)
                data_log["hover_throttle"].append([self.hover_throttle])
                data_log["fa"].append(fa.tolist())
                data_log["pwm"].append(rs)
                data_log["cmd"].append([thr, r_des, p_des, y_des])

            self.prev_vel = cur_vel[:]
            self.prev_posd = des_pos[:]
            self.simulation_time += self.dt
            time.sleep(max(0.0, self.dt - (time.time()-t0)))

        # 结束
        self.client.landAsync().join()
        self.client.armDisarm(False)
        self.client.enableApiControl(False)

        # 输出
        vehicle = "simpleflight"
        condition = wind_tag if wind_tag else "nowind"
        base = f"{vehicle}_{trajectory_type}_{method}_{condition}"

        out_dir = OUT_DIR_TRAIN if trajectory_type=="random" else OUT_DIR_TEST
        if out_root is not None:
            out_dir = Path(out_root)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_csv = out_dir / f"{base}.csv"

        with open(out_csv, "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["idx","t","p","p_d","v","v_d","q","R","w","T_sp","q_sp","hover_throttle","fa","pwm"])
            for i in range(len(data_log["t"])):
                wr.writerow([
                    i,
                    f"{data_log['t'][i]:.14g}",
                    str(data_log['p'][i]),
                    str(data_log['p_d'][i]),
                    str(data_log['v'][i]),
                    str(data_log['v_d'][i]),
                    str(data_log['q'][i]),
                    str(data_log['R'][i]),
                    str(data_log['w'][i]),
                    str(data_log['T_sp'][i]),
                    str(data_log['q_sp'][i]),
                    str(data_log['hover_throttle'][i]),
                    str(data_log['fa'][i]),
                    str(data_log['pwm'][i]),
                ])
        print(f"飞行数据已保存 {out_csv}")
        self._save_figures(out_dir, base, data_log)
        print(f"飞行图像已保存")

        return data_log

# =============== 轨迹 ===============
def test2(t):
    """8字轨迹（XY 平面）；Z 固定高度"""
    x = 10.0*math.sin(0.5*t)
    y = 10.0*math.sin(0.5*t)*math.cos(0.5*t)
    # z = -5.0
    z = -10.0-2*t
    yaw = 0.0
    return x,y,z,yaw

def test3_random_spline_trajectory():
    """约60s随机样条轨迹"""
    import numpy as np
    from scipy.interpolate import CubicSpline
    #np.random.seed(42)
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

# =============== 重置到原点 ===============
def reset_to_home():
    """彻底复位：清零风 -> reset -> 稍等稳定"""
    c = airsim.MultirotorClient()
    c.confirmConnection()
    try:
        c.simSetWind(airsim.Vector3r(0.0, 0.0, 0.0))
    except Exception:
        pass
    c.reset()
    time.sleep(1.5)  # 等待物理引擎稳定

# =============== 运行入口 ===============
def main():
    import argparse
    parser = argparse.ArgumentParser(description="AirSim adaptive control batch runner")
    parser.add_argument('--mode', choices=['train','test'], default=None, help='train or test (默认全量批跑)')
    parser.add_argument('--idx', type=int, default=None, help='指定索引跑单个风况')
    args = parser.parse_args()

    all_items = list(PROFILES_ALL.values())
    train_list = [(p, WIND_CONDITIONS[p["tag"]]) for p in all_items if WIND_CONDITIONS.get(p["tag"]) in TRAIN_CONDS]
    test_list  = [(p, WIND_CONDITIONS[p["tag"]]) for p in all_items if WIND_CONDITIONS.get(p["tag"]) in TEST_CONDS]
    print(train_list)
    print(test_list)

    def run_one(profile, condition, traj_fn, traj_type, sim_time, out_root=None):
        print(f"[Run] {traj_type} | {condition} | profile={profile['tag']} | {sim_time}s")
        # 每次运行前：强制复位到起飞原点
        reset_to_home()
        # 新建控制器并执行
        ctl = SimpleFlightController()
        ctl.run_simulation(total_time=sim_time,
                           trajectory_func=traj_fn,
                           profile=profile,
                           trajectory_type=traj_type,
                           method="adaptive",
                           wind_tag=condition,
                           out_root=out_root)

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

    # 默认：先全量训练，再全量测试（每组都会复位）
    print("未指定参数，默认批量运行全部训练与测试风况")
    for p,c in train_list:
        run_one(p, c, test3_random_spline_trajectory(), "random", 60.0)  # 训练每组1分钟
    for p,c in test_list:
        run_one(p, c, test2, "fig8", 30.0)  # 测试每组0.5分钟

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[中断] 用户手动停止")
    except Exception as e:
        import traceback
        print(f"[错误] {e}")
        traceback.print_exc()
