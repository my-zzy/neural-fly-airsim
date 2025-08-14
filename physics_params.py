# physics_params.py
import torch
import math

def get_uav_physics(device="cpu", dtype=torch.float32):
    """
    计算 UAV 物理参数（包含 drag_box）
    :param device: torch 设备 ('cpu' 或 'cuda')
    :param dtype: torch 数据类型
    :return: dict，包括 drag_box, inertia, max_thrust 等参数
    """

    # 无人机基本参数
    UAV_mass = 1.0  # kg
    UAV_arm_length = 0.2275  # m
    UAV_rotor_z_offset = 0.025  # m

    # 电机/螺旋桨参数
    UAV_rotor_C_T = 0.109919  # 推力系数
    UAV_rotor_C_P = 0.040164  # 扭矩系数
    air_density = 1.225  # kg/m³
    UAV_rotor_max_rpm = 6396.667
    UAV_propeller_diameter = 0.2286  # m
    UAV_propeller_height = 0.01  # m
    UAV_tc = 0.005  # 电机滤波时间常数

    # 最大推力 & 扭矩
    UAV_max_thrust = 4.179446268  # 单电机最大推力 (N)
    UAV_max_torque = 0.055562  # 单电机最大扭矩 (N·m)

    # 阻力系数
    UAV_linear_drag_coefficient = 0.325
    UAV_angular_drag_coefficient = 0.0

    # 机身质量比例
    UAV_body_mass_fraction = 0.78
    UAV_body_mass = UAV_mass * UAV_body_mass_fraction
    UAV_motor_mass = UAV_mass * (1 - UAV_body_mass_fraction) / 4.0

    # 机身盒子尺寸
    UAV_dim_x = 0.180
    UAV_dim_y = 0.110
    UAV_dim_z = 0.040

    # 转动惯量计算
    Ixx_body = UAV_body_mass / 12.0 * (UAV_dim_y**2.0 + UAV_dim_z**2.0)
    Iyy_body = UAV_body_mass / 12.0 * (UAV_dim_x**2.0 + UAV_dim_z**2.0)
    Izz_body = UAV_body_mass / 12.0 * (UAV_dim_x**2.0 + UAV_dim_y**2.0)

    L_eff_sq = (UAV_arm_length * math.cos(math.pi / 4.0))**2.0
    rotor_z_dist_sq = UAV_rotor_z_offset**2.0

    Ixx_motors = 4 * UAV_motor_mass * (L_eff_sq + rotor_z_dist_sq)
    Iyy_motors = 4 * UAV_motor_mass * (L_eff_sq + rotor_z_dist_sq)
    Izz_motors = 4 * UAV_motor_mass * (2.0 * L_eff_sq)

    inertia_diag = torch.tensor(
        [Ixx_body + Ixx_motors, Iyy_body + Iyy_motors, Izz_body + Izz_motors],
        device=device, dtype=dtype
    )

    # 气动阻力计算
    UAV_xy_area = UAV_dim_x * UAV_dim_y + 4.0 * math.pi * UAV_propeller_diameter**2
    UAV_yz_area = UAV_dim_y * UAV_dim_z + 4.0 * math.pi * UAV_propeller_diameter * UAV_propeller_height
    UAV_xz_area = UAV_dim_x * UAV_dim_z + 4.0 * math.pi * UAV_propeller_diameter * UAV_propeller_height

    drag_box = 0.5 * UAV_linear_drag_coefficient * torch.tensor(
        [UAV_yz_area, UAV_xz_area, UAV_xy_area],
        device=device, dtype=dtype
    )

    return {
        "mass": UAV_mass,
        "arm_length": UAV_arm_length,
        "rotor_C_T": UAV_rotor_C_T,
        "rotor_C_P": UAV_rotor_C_P,
        "air_density": air_density,
        "rotor_max_rpm": UAV_rotor_max_rpm,
        "propeller_diameter": UAV_propeller_diameter,
        "tc": UAV_tc,
        "max_thrust": UAV_max_thrust,
        "max_torque": UAV_max_torque,
        "inertia_diag": inertia_diag,
        "drag_box": drag_box
    }


if __name__ == "__main__":
    params = get_uav_physics()
    print("UAV drag_box:", params["drag_box"])
    print("UAV inertia:", params["inertia_diag"])
