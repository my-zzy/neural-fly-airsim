import numpy as np
import torch


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"

# 自适应控制参数 - Conservative tuning for stability
cx = 3
cy = 3
cz = 0.1
cu = 2
cv = 2
cw = 10
lamx = 0.2
lamy = 0.2
lamz = 1

lamphi = 20
lamthe = 20
lampsi = 8
cphi = 17
cthe = 17
cpsi = 10
cp = 0.1
cq = 0.1
cr = 0.1
lamphi_star = 0.1
lamthe_star = 0.1
lampsi_star = 0.1

# 无人机动力学模型参数
UAV_mass=1.0 # 无人机总重量
UAV_arm_length = 0.2275 # 无人机臂长度
UAV_rotor_z_offset = 0.025 # 电机高度

# 电机参数与空气密度
UAV_rotor_C_T = 0.109919 # 螺旋桨推力系数
UAV_rotor_C_P = 0.040164 # 螺旋桨扭矩系数
air_density = 1.225 # 空气密度
UAV_rotor_max_rpm = 6396.667 # 电机最大转速
UAV_propeller_diameter = 0.2286 # 螺旋桨直径
UAV_propeller_height = 0.01 # 螺旋桨截面高度
UAV_tc = 0.005 # 无人机电机滤波时间常数, 越大滤波效果越明显
UAV_max_thrust = 4.179446268 # 无人机单电机最大推力
UAV_max_torque = 0.055562 # 无人机单电机最大扭矩
UAV_linear_drag_coefficient = 0.325 # 线阻力系数
UAV_angular_drag_coefficient = 0.0 # 角阻力系数；有值的时候每个DT都会导致最后变成推力矩与阻力矩平衡，无人机y方向角速度锁定在0.02}
UAV_body_mass_fraction = 0.78 # 无人机中心盒重量占比
UAV_body_mass = UAV_mass * UAV_body_mass_fraction # 无人机中心盒质量
UAV_motor_mass = UAV_mass * (1-UAV_body_mass_fraction) / 4.0 # 电机质量
UAV_dim_x = 0.180; UAV_dim_y = 0.110; UAV_dim_z = 0.040 # 机身盒尺寸
Ixx_body = UAV_body_mass / 12.0 * (UAV_dim_y**2.0 + UAV_dim_z**2.0) # 机身对三个轴的转动惯量
Iyy_body = UAV_body_mass / 12.0 * (UAV_dim_x**2.0 + UAV_dim_z**2.0)
Izz_body = UAV_body_mass / 12.0 * (UAV_dim_x**2.0 + UAV_dim_y**2.0)
L_eff_sq = (UAV_arm_length * torch.cos(torch.tensor(torch.pi / 4.0)))**2.0 # 电机位置偏移量的平方
rotor_z_dist_sq = UAV_rotor_z_offset**2.0 # 电机高度偏移量
Ixx_motors = 4 * UAV_motor_mass * (L_eff_sq + rotor_z_dist_sq)
Iyy_motors = 4 * UAV_motor_mass * (L_eff_sq + rotor_z_dist_sq)
Izz_motors = 4 * UAV_motor_mass * (2.0 * L_eff_sq)
UAV_inertia_diag = torch.tensor([ # 转动惯量矩阵
            Ixx_body + Ixx_motors,
            Iyy_body + Iyy_motors,
            Izz_body + Izz_motors
        ], device=device, dtype=torch.float32)
UAV_xy_area = UAV_dim_x * UAV_dim_y + 4.0 * torch.pi * UAV_propeller_diameter**2
UAV_yz_area = UAV_dim_y * UAV_dim_z + 4.0 * torch.pi * UAV_propeller_diameter * UAV_propeller_height
UAV_xz_area = UAV_dim_x * UAV_dim_z + 4.0 * torch.pi * UAV_propeller_diameter * UAV_propeller_height
drag_box = 0.5 * UAV_linear_drag_coefficient * torch.tensor([UAV_yz_area, UAV_xz_area, UAV_xy_area], device=device, dtype=torch.float32) # 三轴阻力系数“盒”

# 补全电机参数
# UAV_revolutions_per_second = UAV_rotor_max_rpm / 60.0
# UAV_max_speed_rad_s = UAV_revolutions_per_second * 2 * np.pi
# UAV_max_speed_sq = UAV_max_speed_rad_s**2
# UAV_max_thrust = UAV_rotor_C_T * air_density * (UAV_revolutions_per_second**2) * (UAV_propeller_diameter**4) # 电机最大推力计算
# UAV_max_torque = UAV_rotor_C_P * air_density * (UAV_revolutions_per_second**2) * (UAV_propeller_diameter**5) / (2 * np.pi) # 电机最大扭矩计算


# pd parameters
kp1 = 0.8
kd1 = 0.2
kp2 = 0.8
kd2 = 0.2
kp3 = 0.8
kd3 = 0.4
kp4 = 50
kd4 = 20
kp5 = 20
kd5 = 15
kp6 = 60
kd6 = 30


'''
# 仿真与通用参数
DT = 0.10  # MPC轨迹每步步长
"""在GPU并行生成轨迹处理之后N=100时计算时间减少到了0.005s以下
   N=1000时为0.005-0.007s
   N=10000时为0.009s左右
   网络每次train耗时大约0.003s"""
MAX_SIM_TIME_PER_EPISODE = 30  # 单个Episode最大时间
NUM_EPISODES = 1000  # 训练最大episode数
POS_TOLERANCE = 1  # 判定抵达目标的位置误差限 (meters)
VELO_TOLERANCE = 1  # 判定抵达目标的速度误差限 (m/s)
CONTROL_MAX = 0.66 # 最大控制指令范围（simpleflight为速度范围，PX4为加速度，现在是油门信号）
CONTROL_MIN = 0.62 # 油门信号下限

# CEM参数
PREDICTION_HORIZON = 5  # MPC预测长度 (N_steps)
N_SAMPLES_CEM = 200000  # 每个CEM采样过程采样数量
N_ELITES_CEM = int(0.1 * N_SAMPLES_CEM)  # CEM精英群体数量
N_ITER_CEM = 1  # 每个MPC优化步的CEM迭代轮数
INITIAL_STD_CEM = CONTROL_MAX  # CEM采样初始标准差，给大一点利于探索
MIN_STD_CEM = 0.05  # CEM标准差最小值
ALPHA_CEM = 0.8  # CEM方差均值更新时软参数，新的值所占比重

# 神经网络与训练参数
STATE_DIM = 13  # 状态向量维度
ACTION_DIM = 4  # 动作向量维度
NN_HIDDEN_SIZE = 128  # 隐藏层大小
LEARNING_RATE = 5e-5  # 学习率
BUFFER_SIZE = 100000  # buffer大小
BATCH_SIZE = 64  # 训练batch size
NN_TRAIN_EPOCHS_PER_STEP = 5  # 每次训练时训练epoch数
MIN_BUFFER_FOR_TRAINING = BATCH_SIZE  # 开始训练时buffer最小容量
EPISODE_EXPLORE = 5  # 随机探索episode数
SCALER_REFIT_FREQUENCY = 10  # 归一化参数更新频率
FIT_SCALER_SUBSET_SIZE = 2000  # 用于更新归一化参数的样本数

# 穿门任务专用参数
WAYPOINT_PASS_THRESHOLD_Y = -0.5  # 判定无人机穿门的阈值，负值拟合到达门前一点然后冲过去

# PyTorch设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # 代价权重矩阵
# # 控制代价矩阵
# R_CONTROL_COST_NP = R_CONTROL_COST_NP = np.diag([
#     1000,  # FR电机控制量
#     1000,  # RL控制量
#     1000,  # FL控制量
#     1000   # RR控制量
# ])
# R_CONTROL_COST_MATRIX_GPU = torch.tensor(R_CONTROL_COST_NP, dtype=torch.float32, device=device)

# # 运行状态代价矩阵
# # 索引：0-2: 位置 (x,y,z), 3-5: 速度 (vx,vy,vz), 6-8: 姿态 (p,r,y), 9-11: 角速度 (wx,wy,wz)
# Q_STATE_COST_NP = np.diag([
#     2500000.0, 10000.0, 100000.0,  # x,y,z位置
#     1000000.0, 100000.0, 1000000.0,   # x,y,z速度
#     100000.0, 1000000.0, 1000000.0, 1000000.0,     # 姿态
#     1000000.0, 100000.0, 1000000.0      # 角速度
# ])
# Q_STATE_COST_MATRIX_GPU = torch.tensor(Q_STATE_COST_NP, dtype=torch.float32, device=device)

# # 终端状态代价矩阵
# Q_TERMINAL_COST_NP = np.diag([
#     2500000.0, 10000.0, 100000.0,  # x,y,z位置
#     1000000.0, 100000.0, 1000000.0,   # x,y,z速度
#     100000.0, 1000000.0, 1000000.0, 1000000.0,     # 姿态
#     1000000.0, 100000.0, 1000000.0      # 角速度
# ])
# Q_TERMINAL_COST_MATRIX_GPU = torch.tensor(Q_TERMINAL_COST_NP, dtype=torch.float32, device=device)

# 减0后的代价权重矩阵
# 控制代价矩阵
R_CONTROL_COST_NP = np.diag([
    0.1,  # FR电机控制量
    0.1,  # RL控制量
    0.1,  # FL控制量
    0.1   # RR控制量
])
R_CONTROL_COST_MATRIX_GPU = torch.tensor(R_CONTROL_COST_NP, dtype=torch.float32, device=device)

# 运行状态代价矩阵
# 索引：0-2: 位置 (x,y,z), 3-5: 速度 (vx,vy,vz), 6-9: 姿态 (p,r,y), 10-12: 角速度 (wx,wy,wz)
Q_STATE_COST_NP = np.diag([
    250.0, 1.0, 10.0,  # x,y,z位置
    100.0, 10.0, 100.0,   # x,y,z速度
    10.0, 100.0, 100.0, 100.0,     # 姿态
    100.0, 10.0, 100.0      # 角速度
])
Q_STATE_COST_MATRIX_GPU = torch.tensor(Q_STATE_COST_NP, dtype=torch.float32, device=device)

# 终端状态代价矩阵
Q_TERMINAL_COST_NP = np.diag([
    250.0, 1.0, 10.0,  # x,y,z位置
    100.0, 10.0, 100.0,   # x,y,z速度
    10.0, 100.0, 100.0, 100.0,     # 姿态
    100.0, 10.0, 100.0      # 角速度
])
Q_TERMINAL_COST_MATRIX_GPU = torch.tensor(Q_TERMINAL_COST_NP, dtype=torch.float32, device=device)

# 静态代价权重矩阵
# 静态目标控制代价矩阵
STATIC_R_CONTROL_COST_NP = np.diag([
    0.1,  # FR电机控制量
    0.1,  # RL控制量
    0.1,  # FL控制量
    0.1   # RR控制量
])
STATIC_R_CONTROL_COST_MATRIX_GPU = torch.tensor(STATIC_R_CONTROL_COST_NP, dtype=torch.float32, device=device)

# 静态目标运行状态代价矩阵
# 索引：0-2: 位置 (x,y,z), 3-5: 速度 (vx,vy,vz), 6-9: 姿态 (p,r,y), 10-12: 角速度 (wx,wy,wz)
STATIC_Q_STATE_COST_NP = np.diag([
    5.0, 5.0, 10.0,  # x,y,z位置
    2.0, 2.0, 10.0,   # x,y,z速度
    5.0, 10.0, 10.0, 10.0,     # 姿态
    10.0, 5.0, 10.0      # 角速度
])
STATIC_Q_STATE_COST_MATRIX_GPU = torch.tensor(STATIC_Q_STATE_COST_NP, dtype=torch.float32, device=device)

# 静态目标终端状态代价矩阵
STATIC_Q_TERMINAL_COST_NP = np.diag([
    200.0, 150.0, 500.0,  # x,y,z位置
    20.0, 20.0, 150.0,   # x,y,z速度
    50.0, 100.0, 100.0, 100.0,     # 姿态
    100.0, 50.0, 100.0      # 角速度
])
STATIC_Q_TERMINAL_COST_MATRIX_GPU = torch.tensor(STATIC_Q_TERMINAL_COST_NP, dtype=torch.float32, device=device)

# AirSim参数
door_frames_names = ["men_Blueprint", "men2_Blueprint"]
door_param= { # 门的正弦运动参数
            "amplitude": 2,  # 运动幅度（米）
            "frequency": 0.1,  # 运动频率（Hz）
            "deviation": None,  # 两个门的初始相位 (set in reset)
            "initial_x_pos": None,  # 门的初始x位置 (set in reset)
            "start_time":None # 门运动的初始时间
        }

'''