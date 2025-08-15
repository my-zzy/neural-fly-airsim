#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sys, csv, time, json, math, select, threading
from pathlib import Path
import numpy as np
import airsim
import matplotlib.pyplot as plt

# ========= 采集参数 =========
OUT_DIR = Path("logs_manual_wind_profiles_rc")
DT = 0.02                   # 50 Hz
SIM_TIME_PER_PROFILE = None # None=回车开始/按键结束；否则固定秒数自动结束
MASS_KG = 1.0
G = 9.81
HOVER_THROTTLE = 0.44
USE_SELF_AS_REFERENCE = True

# 起飞检测/预热
TSP_START_THRESH = 0.05     # 只有当 RC 油门 > 5% 才认为要起飞
ROTOR_W_START = 30.0        # 平均桨速 > 30 rad/s 视为转起来
ALT_DELTA_START = 0.10      # 高度变化 > 0.10 m 也视为起飞（|Δz|>0.1, NED）
WARMUP_SKIP_FRAMES = 10     # 真正开始记录前跳过的帧数（滤掉瞬态）

# ========= 风场预设 =========
PROFILES = [
    {"tag":"0mps",   "kind":"const", "dir":(1,0,0), "mag":0.0},
    {"tag":"4.2mps", "kind":"const", "dir":(1,0,0), "mag":4.2},
    {"tag":"8.5mps", "kind":"const", "dir":(1,0,0), "mag":8.5},
    {"tag":"12.1mps","kind":"const", "dir":(1,0,0), "mag":12.1},
    {"tag":"sinusoidal_0to12mps","kind":"sin","dir":(1,0,0),"mag_mean":6.0,"mag_amp":6.0,"freq_hz":0.25}
]

# ========= 工具 =========
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

def compute_fa_from_Tsp(m, a_world, R_b2w, T_sp):
    T = (T_sp / HOVER_THROTTLE) * m * G      # 线性标定
    f_b = np.array([0.0, 0.0, T])            # +Z_b 向下(NED)
    fT_world = R_b2w @ f_b
    e3 = np.array([0.0, 0.0, 1.0])
    return m*a_world - fT_world - m*G*e3

# --- 跨平台“回车开始/任意键结束” ---
def wait_enter_or_timeout(prompt=">>> 回车开始该档采集（或等待倒计时）...", timeout=5.0):
    print(f"{prompt}  {int(timeout)}s")
    done = {"v": False}
    def reader():
        try: input()
        except Exception: pass
        done["v"] = True
    th = threading.Thread(target=reader, daemon=True); th.start()
    t0 = time.time()
    while time.time() - t0 < timeout:
        if done["v"]: return
        time.sleep(0.1)

def keypressed_nonblock():
    try:
        import msvcrt
        return msvcrt.kbhit()
    except Exception:
        try:
            r,_,_ = select.select([sys.stdin], [], [], 0.0)
            return bool(r)
        except Exception:
            return False

def flush_enter():
    try:
        import msvcrt
        while msvcrt.kbhit(): _ = msvcrt.getwch()
    except Exception:
        try:
            if sys.stdin.readable(): _ = sys.stdin.readline()
        except Exception: pass

# =========从 RC 读取 T_sp =========
def read_Tsp_from_rc(client):
    """
    读取 RC 油门并映射到 [0,1]：
    - 常见返回 [0,1]；也可能为 [-1,1] 或 1000~2000 us。
    """
    try:
        rc = client.getRCData()
        thr_raw = float(getattr(rc, "throttle", float("nan")))
        if not np.isfinite(thr_raw): return None
        # [0,1]
        if 0.0 - 1e-6 <= thr_raw <= 1.0 + 1e-6:
            T_sp = thr_raw
        # [-1,1]
        elif -1.0 - 1e-6 <= thr_raw <= 1.0 + 1e-6:
            T_sp = (thr_raw + 1.0) * 0.5
        # 1000~2000 (us)
        else:
            T_sp = (thr_raw - 1000.0) / 1000.0
        return float(np.clip(T_sp, 0.0, 1.0))
    except Exception:
        return None

# def mean_rotor_speed(client):
#     rs = []
#     rotor_states = client.getRotorStates()
#     for i in range(4):  # 只取前 4 个桨
#         rs.append(rotor_states.rotors[i]['speed'])
#     return rs   

def landed_state_str(s):
    try:
        return "Flying" if int(s.landed_state)==0 else "Landed"
    except Exception:
        return "Unknown"

# ========= 等待起飞：电机转、飞行状态、油门阈值、或高度变化 =========
def wait_until_flying(client, max_wait_s=60.0):
    """
    条件任一满足即通过：
    1) RC 油门 > TSP_START_THRESH
    2) 平均桨速 > ROTOR_W_START
    3) landed_state == Flying
    4) |z - z0| > ALT_DELTA_START
    """
    print(">>> 等待起飞条件满足（油门>阈值 / 桨速>阈值 / Flying / 高度变化>阈值）…")
    t0 = time.time()
    z0 = client.getMultirotorState().kinematics_estimated.position.z_val
    last_print = 0.0

    while True:
        now = time.time()
        st = client.getMultirotorState()
        z = st.kinematics_estimated.position.z_val
        rc_tsp = read_Tsp_from_rc(client)
        rc_tsp_val = rc_tsp if rc_tsp is not None else 0.0
        #wmean = mean_rotor_speed(client)
        flying = (landed_state_str(st) == "Flying")
        dz = abs(z - z0)

        # 状态打印（
        if (rc_tsp_val > TSP_START_THRESH)  or flying or (dz > ALT_DELTA_START):
            print(">>> 起飞条件满足，开始记录（预热若干帧后写入）")
            return True

        if now - t0 > max_wait_s:
            print(">>> 等待超时，仍未检测到起飞；将开始记录（可能仍为怠机段）")
            return False

        time.sleep(0.05)

# ========= 主流程 =========
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR/"profiles_meta.json","w") as f: json.dump(PROFILES, f, indent=2)

    client = airsim.MultirotorClient(); client.confirmConnection()

    # 3D 轨迹
    P_all, seg_idx = [], []
    csv_paths = []

    print("\n===== 手柄/人工飞行（RC 读取 T_sp，含“起飞检测闸门”）多风况采集 =====\n"
          "提示：UE 窗口要置前获得焦点；settings.json 里确认 RC.RemoteControlID 指向你的手柄。\n")

    for k, prof in enumerate(PROFILES, 1):
        tag, kind = prof["tag"], prof["kind"]
        d = np.array(prof["dir"], dtype=float); d = d/(np.linalg.norm(d)+1e-12)
        mag = float(prof["mag"]) if kind=="const" else float(prof["mag_mean"])
        wvec = d * mag
        client.simSetWind(airsim.Vector3r(*wvec.tolist()))
        print(f"[{k}/{len(PROFILES)}] 风况: {tag} , 初始 {wvec.tolist()} (m/s)")

        # 开始/结束方式
        if SIM_TIME_PER_PROFILE is None:
            wait_enter_or_timeout(timeout=5.0)
            duration = float('inf')
        else:
            print(f">>> 将采集 {SIM_TIME_PER_PROFILE:.1f}s，开始飞行...")
            duration = SIM_TIME_PER_PROFILE

        # ==== 等待起飞 ====
        wait_until_flying(client, max_wait_s=60.0)

        # CSV
        csv_path = OUT_DIR / f"manual_{tag}.csv"; csv_paths.append(csv_path)
        with open(csv_path, "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["t","p","p_d","v","v_d","q","R","w","T_sp","q_sp",
                         "hover_throttle","fa","pwm"])

            t0 = time.time(); t_prev = t0; v_prev = None
            warmup = WARMUP_SKIP_FRAMES   # 预热帧计数

            while True:
                now = time.time()
                # 正弦阵风实时更新
                if kind == "sin":
                    t_rel = now - t0
                    mag_t = prof["mag_mean"] + prof["mag_amp"] * math.sin(2*math.pi*prof["freq_hz"]*t_rel)
                    mag_t = max(0.0, float(mag_t))
                    wvec = d * mag_t
                    client.simSetWind(airsim.Vector3r(*wvec.tolist()))

                # 固定步进
                if now - t_prev < DT: time.sleep(DT - (now - t_prev))
                t_prev = time.time(); t = t_prev - t0

                st = client.getMultirotorState()
                kin = st.kinematics_estimated
                p = np.array(v3r(kin.position))
                v = np.array(v3r(kin.linear_velocity))
                omg = np.array(v3r(kin.angular_velocity))
                q = kin.orientation
                qw,qx,qy,qz = q.w_val,q.x_val,q.y_val,q.z_val
                R = quat_wxyz_to_R(qw,qx,qy,qz)

                # 参考：自身或 NaN
                if USE_SELF_AS_REFERENCE:
                    p_d, v_d, q_sp = p, v, [qx,qy,qz,qw]
                else:
                    p_d, v_d, q_sp = [float('nan')]*3, [float('nan')]*3, [float('nan')]*4

                # ===== 仅用方法1：RC 读取 T_sp =====
                T_sp = read_Tsp_from_rc(client)
                if T_sp is None:
                    T_sp = 0.0  # 兜底：避免 NaN

                # 加速度（数值微分）
                if v_prev is None: a_world = np.zeros(3)
                else: a_world = (v - v_prev) / DT
                v_prev = v.copy()

                # fa
                fa = compute_fa_from_Tsp(MASS_KG, a_world, R, T_sp)

                # 电机转速（当作 pwm；SimpleFlight 可用）
                rs = []
                rotor_states = client.getRotorStates()
                for i in range(4):  # 只取前 4 个桨
                    rs.append(rotor_states.rotors[i]['speed'])
                pwm_list = rs

                # 预热帧：不写入，只消耗
                if warmup > 0:
                    warmup -= 1
                else:
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
                        to_json_arr(q_sp),
                        to_json_arr([HOVER_THROTTLE]),
                        to_json_arr(fa),
                        to_json_arr(pwm_list),
                    ])
                    P_all.append(p.tolist()); seg_idx.append(k-1)

                # 结束条件
                if SIM_TIME_PER_PROFILE is None:
                    if keypressed_nonblock():
                        flush_enter()
                        print(">>> 按键已检测到，结束本档。")
                        break
                else:
                    if t >= duration - 1e-9:
                        print(">>> 该档时间到，自动结束。")
                        break

        print(f"[OK] 保存：{csv_path}\n")

    # 合并
    merged = OUT_DIR / "dataset_all_profiles.csv"
    with open(merged, "w", newline="") as fout:
        writer = None
        for pth in csv_paths:
            with open(pth, "r") as fin:
                rd = csv.reader(fin); head = next(rd)
                if writer is None: writer = csv.writer(fout); writer.writerow(head)
                for row in rd: writer.writerow(row)
    print(f"[OK] 合并：{merged}")

    # 3D 轨迹（分段着色）
    if len(P_all) > 1:
        P = np.array(P_all); seg = np.array(seg_idx)
        colors = ["#444444","#1f77b4","#2ca02c","#d62728","#9467bd"]
        fig = plt.figure(figsize=(6,5))
        ax = fig.add_subplot(111, projection="3d")
        for i, prof in enumerate(PROFILES):
            idx = (seg==i)
            if np.any(idx):
                ax.plot(P[idx,0], P[idx,1], P[idx,2], lw=1.6, color=colors[i%len(colors)], label=prof["tag"])
        ax.set_xlabel("X (m, N)"); ax.set_ylabel("Y (m, E)"); ax.set_zlabel("Z (m, Down)")
        ax.set_title("Manual flight 3D trajectory by wind profile (RC T_sp, takeoff-gated)")
        ax.legend(loc="best", fontsize=8)
        plt.tight_layout()
        png = OUT_DIR/"traj3d_by_profile.png"
        plt.savefig(png, dpi=180); plt.show()
        print(f"[OK] 3D 轨迹图：{png}")

if __name__ == "__main__":
    main()
