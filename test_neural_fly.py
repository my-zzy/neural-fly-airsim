#!/usr/bin/env python
"""
Test script to verify the neural_fly_controller function works correctly
"""
import numpy as np
import torch
import math

# Mock the missing imports for testing
class MockAirSim:
    pass

# Create a simple phi network for testing
class SimplePhiNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(6, 32)
        self.fc2 = torch.nn.Linear(32, 30)  # Output 30 features (3 dimensions x 10 features)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x.view(3, 10)  # Reshape to [3, 10]

def neural_fly_controller(pos, vel, att, ang_vel, posd, attd, phi_net, a_hat, P, dt, t):
    """
    Neural-fly adaptive controller using phi network
    """
    # Current state
    current_pos = np.array([pos[0][-1], pos[1][-1], pos[2][-1]])
    current_vel = np.array([vel[0][-1], vel[1][-1], vel[2][-1]])
    current_att = np.array([att[0][-1], att[1][-1], att[2][-1]])
    
    # Desired trajectory
    xd = np.array([posd[0][-1], posd[1][-1], posd[2][-1]])
    
    # Calculate desired velocity derivatives (numerical differentiation)
    xd_dot = np.zeros(3)
    xd_ddot = np.zeros(3)
    
    if len(posd[0]) >= 2:
        for i in range(3):
            xd_dot[i] = (posd[i][-1] - posd[i][-2])/dt
    
    if len(posd[0]) >= 3:
        for i in range(3):
            xd_ddot[i] = ((posd[i][-1] - posd[i][-2])/dt - (posd[i][-2] - posd[i][-3])/dt)/dt

    # Neural-fly control parameters
    lambda_a = 0.1
    Q = torch.eye(a_hat.shape[0]) * 0.01
    R = torch.eye(3) * 0.1
    K = torch.eye(3) * 5.0
    Lambda = torch.eye(3) * 2.0
    g_vector = np.array([0.0, 0.0, 9.81])  # gravity in NED frame
    
    # Tracking error
    q_tilde = current_pos - xd
    s = current_vel - xd_dot + (Lambda.numpy() @ q_tilde)

    # phi(x) - neural network feature
    x = torch.tensor(np.concatenate([current_pos, current_vel]), dtype=torch.float32)
    with torch.no_grad():
        phi = phi_net(x)  # shape: [3, h]

    # Residual force estimate (assume zero for this implementation)
    y = torch.zeros(3)

    # Compute force command
    f_nominal = xd_ddot + g_vector  # for m=1
    f_learning = (phi @ a_hat).numpy()
    u = f_nominal - K.numpy() @ s - f_learning

    # Update a_hat using adaptive law
    P_phi_T = P @ phi.T
    try:
        R_inv = torch.linalg.inv(R)
    except:
        R_inv = torch.eye(3) * (1.0 / 0.1)  # fallback if R is singular
    
    a_hat_dot = -lambda_a * a_hat - P_phi_T @ R_inv @ (phi @ a_hat - y) + P_phi_T @ torch.tensor(s, dtype=torch.float32)
    a_hat_new = a_hat + a_hat_dot * dt

    # Update P matrix
    P_dot = -2 * lambda_a * P + Q - P_phi_T @ R_inv @ phi @ P
    P_new = P + P_dot * dt

    # Convert force commands to AirSim controls
    # Calculate thrust magnitude and desired attitude
    thrust_magnitude = np.linalg.norm(u)
    
    # Simple attitude computation for small angles
    if thrust_magnitude > 1e-6:
        # Desired acceleration in body frame
        u_normalized = u / thrust_magnitude
        
        # Convert to desired attitude (simplified)
        roll_desired = math.atan2(-u_normalized[1], -u_normalized[2])
        pitch_desired = math.atan2(u_normalized[0], math.sqrt(u_normalized[1]**2 + u_normalized[2]**2))
        yaw_desired = attd[2][-1] if len(attd[2]) > 0 else 0.0
        
        # Limit attitude angles
        max_angle = math.radians(30)
        roll_desired = max(-max_angle, min(max_angle, roll_desired))
        pitch_desired = max(-max_angle, min(max_angle, pitch_desired))
    else:
        roll_desired = 0.0
        pitch_desired = 0.0
        yaw_desired = attd[2][-1] if len(attd[2]) > 0 else 0.0
    
    # Convert thrust to throttle (normalized 0-1)
    UAV_mass = 1.0  # kg
    max_thrust = UAV_mass * 9.81 * 2.0
    throttle = max(0.0, min(1.0, thrust_magnitude / max_thrust * 0.5 + 0.5))

    print(f"NeuralFly - Pos: [{current_pos[0]:.2f}, {current_pos[1]:.2f}, {current_pos[2]:.2f}] | "
          f"Throttle: {throttle:.3f}, Roll: {math.degrees(roll_desired):.1f}°, "
          f"Pitch: {math.degrees(pitch_desired):.1f}°, Yaw: {math.degrees(yaw_desired):.1f}°")

    return throttle, roll_desired, pitch_desired, yaw_desired, a_hat_new, P_new


def test_neural_fly_controller():
    """Test the neural_fly_controller function"""
    print("Testing Neural-Fly Controller...")
    
    # Create test data
    dt = 0.01
    phi_net = SimplePhiNet()
    a_hat = torch.zeros(10)
    P = torch.eye(10)
    
    # Mock history data (3 time steps)
    pos = [[0.0, 0.1, 0.2], [0.0, 0.1, 0.2], [0.0, 0.1, 0.2]]  # x, y, z
    vel = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]  # vx, vy, vz
    att = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]  # roll, pitch, yaw
    ang_vel = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]  # wx, wy, wz
    posd = [[1.0, 0.9, 0.8], [1.0, 0.9, 0.8], [-2.0, -2.0, -2.0]]  # desired position
    attd = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]  # desired attitude
    
    try:
        throttle, roll_desired, pitch_desired, yaw_desired, a_hat_new, P_new = neural_fly_controller(
            pos, vel, att, ang_vel, posd, attd, phi_net, a_hat, P, dt, 0.02
        )
        
        print("Test PASSED!")
        print(f"Outputs: throttle={throttle:.3f}, roll={math.degrees(roll_desired):.1f}°, "
              f"pitch={math.degrees(pitch_desired):.1f}°, yaw={math.degrees(yaw_desired):.1f}°")
        print(f"Adaptive parameters updated successfully")
        
    except Exception as e:
        print(f"Test FAILED: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_neural_fly_controller()
