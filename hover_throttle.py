#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AirSim SimpleFlight 悬停油门标定 (自动悬停 + 电机转速采样 + 平均滤波)
"""

import airsim
import time
import numpy as np
import math

# ====== 无人机参数 ======
mass = 1.0                 # kg
g = 9.81                   # m/s^2
CT = 0.109919              # 螺旋桨推力系数
rho = 1.225                # 空气密度
D = 0.2286                 # 螺旋桨直径 (m)
max_rpm = 6396.667         # 电机最大转速 (rpm)

# ====== 采样参数 ======
HOVER_ALT = -5.0           # 悬停高度 (m, NED坐标，负值表示向上)
SAMPLE_TIME = 5.0          # 采样总时长 (s)
SAMPLE_DT = 0.1            # 采样间隔 (s)

# ====== 初始化 ======
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

print("Takeoff...")
client.takeoffAsync().join()
client.moveToZAsync(HOVER_ALT, 1).join()
time.sleep(2.0)  # 等待稳定

# ====== 开始采样 ======
print("\n开始采样电机转速...")
rpm_samples = []

t0 = time.time()
while time.time() - t0 < SAMPLE_TIME:
    rotor_states = client.getRotorStates()
    rs = []
    for i in range(4):  # 四个电机
        omega = rotor_states.rotors[i]['speed']  # rad/s
        rpm = omega * 60.0 / (2*math.pi)        # 转换成 rpm
        rs.append(rpm)
    rpm_mean = np.mean(rs)
    rpm_samples.append(rpm_mean)
    print(f"采样: 平均电机转速 = {rpm_mean:.2f} rpm")
    time.sleep(SAMPLE_DT)

# ====== 结果计算 ======
mean_rpm = np.mean(rpm_samples)
print("\n==========================")
print(f"悬停平均电机转速 = {mean_rpm:.2f} rpm")

# 单电机推力 (理论计算)
n = mean_rpm / 60.0  # 转/秒
T_motor = CT * rho * (n**2) * (D**4)
T_total = T_motor * 4
print(f"估算总推力 = {T_total:.3f} N (理论重力 {mass*g:.3f} N)")

# 悬停油门估算（假设推力 ~ (转速/max)^2）
hover_throttle = (mean_rpm / max_rpm)**2
print(f"估算 hover_throttle ≈ {hover_throttle:.3f}")
print("==========================\n")

# ====== 降落 ======
print("Landing...")
client.landAsync().join()
client.armDisarm(False)
client.enableApiControl(False)
