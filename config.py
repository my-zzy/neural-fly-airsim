import numpy as np
import torch


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"

# 自适应控制参数 - Conservative tuning for stability
cx = 3
cy = 3
cz = 0.6
cu = 2
cv = 2
cw = 5
lamx = 0.2
lamy = 0.2
lamz = 10

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

# kp1 = 4.0
# kd1 = 2.0
# kp2 = 4.0
# kd2 = 2.0
# kp3 = 1.0
# kd3 = 0.8

# ki1 = 0.2
# ki2 = 0.2
# ki3 = 0.5

kp1 = 3.2    
kd1 = 2.8    
ki1 = 0.1
kp2 = 3.2   
kd2 = 2.8    
ki2 = 0.1
kp3 = 1.0    
kd3 = 0.8     
ki3 = 0.3
