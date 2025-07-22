import airsim
import torch
import numpy as np

# Connect to AirSim
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

# Takeoff
client.takeoffAsync().join()

# === Your trained phi network ===
phi_net = torch.load("trained_phi.pt")
phi_net.eval()

# Gains and parameters
lambda_a = 0.1
P = torch.eye(phi_net.output_shape[1])  # shape: [h,h]
Q = torch.eye(phi_net.output_shape[1]) * 0.01
R = torch.eye(3) * 0.1  # residual force measurement noise
K = torch.eye(3) * 5.0
Lambda = torch.eye(3) * 2.0

# Initial linear coeffs
a_hat = torch.zeros(phi_net.output_shape[1])

# Desired trajectory
def desired_trajectory(t):
    # Figure-8 in X-Y plane at 2 m height
    xd = np.array([2.0 * np.sin(t), 2.0 * np.sin(t) * np.cos(t), -2.0])
    xd_dot = np.array([2.0 * np.cos(t), 2.0 * (np.cos(t)**2 - np.sin(t)**2), 0.0])
    xd_ddot = np.array([-2.0 * np.sin(t), -8.0 * np.sin(t) * np.cos(t), 0.0])
    return xd, xd_dot, xd_ddot

# Control loop
dt = 0.02
for step in range(1000):
    t = step * dt

    # Get state
    state = client.getMultirotorState()
    pos = np.array([state.kinematics_estimated.position.x_val,
                    state.kinematics_estimated.position.y_val,
                    state.kinematics_estimated.position.z_val])
    vel = np.array([state.kinematics_estimated.linear_velocity.x_val,
                    state.kinematics_estimated.linear_velocity.y_val,
                    state.kinematics_estimated.linear_velocity.z_val])

    # Desired trajectory
    xd, xd_dot, xd_ddot = desired_trajectory(t)

    # Tracking error
    q_tilde = pos - xd
    s = vel - xd_dot + Lambda @ q_tilde

    # phi(x)
    x = torch.tensor(np.concatenate([pos, vel]), dtype=torch.float32)
    phi = phi_net(x)  # shape: [3, h]

    # Residual force estimate (assume you have y=measured residual force)
    # Here we can't measure real residual, so assume zero for sketch:
    y = torch.zeros(3)

    # Compute force command
    f_nominal = xd_ddot + g_vector()  # for m=1
    f_learning = phi @ a_hat
    u = f_nominal - K @ torch.tensor(s) - f_learning

    # Update a_hat
    P_phi_T = P @ phi.T
    a_hat_dot = -lambda_a * a_hat - P_phi_T @ torch.linalg.inv(R) @ (phi @ a_hat - y) + P_phi_T @ torch.tensor(s)
    a_hat += a_hat_dot * dt

    # Update P
    P_dot = -2 * lambda_a * P + Q - P_phi_T @ torch.linalg.inv(R) @ phi @ P
    P += P_dot * dt

    # Convert force to AirSim commands (simple mapping)
    thrust = np.linalg.norm(u)
    attitude = compute_attitude_from_force(u.numpy())  # your custom function

    # Send to PX4 via AirSim
    client.moveByRollPitchYawrateThrottleAsync(
        roll=attitude[0],
        pitch=attitude[1],
        yaw_rate=attitude[2],
        throttle=thrust / max_thrust_value  # normalize
    )

    time.sleep(dt)

client.armDisarm(False)
client.enableApiControl(False)
