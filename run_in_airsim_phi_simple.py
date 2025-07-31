#!/usr/bin/env python
import airsim
import time
import numpy as np
import math
import matplotlib.pyplot as plt
import torch
from config import *
from mlmodel import load_model

def quaternion_to_euler(x, y, z, w):
    """
    Convert a quaternion into roll, pitch, yaw (in radians)
    Roll  = rotation around x-axis
    Pitch = rotation around y-axis
    Yaw   = rotation around z-axis
    """
    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = math.copysign(math.pi / 2, sinp)  # use 90 degrees if out of range
    else:
        pitch = math.asin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    # in radians

    return roll, pitch, yaw

def get_quaternion_from_euler(roll, pitch, yaw):
    """
    Convert Euler angles (roll, pitch, yaw) to quaternion (qw, qx, qy, qz)
    """
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy

    return np.array([qw, qx, qy, qz])

def neural_fly_controller(pos, vel, att, ang_vel, posd, attd, phi_net, a_hat, P, dt, t, client):
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
    # x,y,z all in x below
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
    Q = torch.eye(a_hat.shape[0], dtype=torch.float64) * 0.01
    R = torch.eye(3, dtype=torch.float64) * 0.1
    K = torch.eye(3, dtype=torch.float64) * 5.0
    Lambda = torch.eye(3, dtype=torch.float64) * 2.0
    g_vector = np.array([0.0, 0.0, 9.81])  # gravity in NED frame
    
    # Tracking error
    q_tilde = current_pos - xd
    s = current_vel - xd_dot + (Lambda.numpy() @ q_tilde)

    # phi(x) - neural network feature
    # Input: current velocity (3) + quaternion (4) + rotor speeds (4) = 11 total
    # Get quaternion directly from AirSim client
    try:
        state = client.getMultirotorState()
        orientation = state.kinematics_estimated.orientation
        # AirSim quaternion format: [qw, qx, qy, qz]
        quaternion = np.array([orientation.w_val, orientation.x_val, orientation.y_val, orientation.z_val])
    except Exception as e:
        print(f"Warning: Could not get quaternion from client: {e}")
        # Fallback to converting from Euler angles
        quaternion = get_quaternion_from_euler(current_att[0], current_att[1], current_att[2])
    
    # Get rotor speeds from AirSim
    try:
        rotor_states = client.getRotorStates()
        rotor_speeds = [rotor_states.rotors[i]['speed']/1000 for i in range(4)]
        # in utils.py line 96 rotorspeed is divided by 1000
    except Exception as e:
        print(f"Warning: Could not get rotor speeds: {e}")
        # Use default/estimated rotor speeds based on hover condition
        hover_rpm = 3000.0  # Approximate hover RPM
        rotor_speeds = [hover_rpm, hover_rpm, hover_rpm, hover_rpm]
    
    # Construct phi network input: [vx, vy, vz, qw, qx, qy, qz, rotor1, rotor2, rotor3, rotor4]
    x = torch.tensor(np.concatenate([current_vel, quaternion, rotor_speeds]), dtype=torch.float64)
    print(f"x: {x}")
    with torch.no_grad():
        phi = phi_net.phi(x)  # shape: [3, h] or [h, 3] depending on network architecture
        # print("phi shape:", phi.shape)  # h = 4
        print(f"phi: {phi}")
        if phi.dim() == 1:  # If phi is 1D, expand it to [3, h]
            phi = phi.unsqueeze(0).repeat(3, 1)
        elif phi.shape[0] != 3:  # If first dimension is not 3, transpose
            phi = phi.T

    # FOR TESTING
    # phi = torch.ones_like(phi)
    # print("x shape:", x.shape)
    # print("altered phi shape:", phi.shape)
    # print("a_hat shape:", a_hat.shape)

    # Residual force estimate (assume zero for this implementation)
    y = torch.zeros(3, dtype=torch.float64)
    # y = phi @ a_hat

    # Compute force command
    f_nominal = xd_ddot + g_vector  # for m=1
    f_learning = (phi @ a_hat).numpy()
    u = f_nominal - K.numpy() @ s - f_learning

    # Update a_hat using adaptive law
    P_phi_T = P @ phi.T
    print(f"P: {P}, phi.t: {phi.T}")
    try:
        R_inv = torch.linalg.inv(R)
    except:
        print("R is singular")
        R_inv = torch.eye(3, dtype=torch.float64) * (1.0 / 0.1)  # fallback if R is singular
    
    print(f"{-lambda_a * a_hat}, {- P_phi_T @ R_inv @ (phi @ a_hat - y)}, {P_phi_T @ torch.tensor(s, dtype=torch.float64)}")
    a_hat_dot = -lambda_a * a_hat - P_phi_T @ R_inv @ (phi @ a_hat - y) + P_phi_T @ torch.tensor(s, dtype=torch.float64)
    a_hat_new = a_hat + a_hat_dot * dt
    # a_hat_new = torch.zeros_like(a_hat_new)

    # Update P matrix
    P_dot = -2 * lambda_a * P + Q - P_phi_T @ R_inv @ phi @ P
    P_new = P + P_dot * dt

    # Convert force commands to AirSim controls
    # Calculate thrust magnitude and desired attitude
    # print(u)
    # u = u / (UAV_mass * 9.81) * 0.5 + 0.5
    thrust_magnitude = np.linalg.norm(u)
    print(f"{f_nominal}, {-K.numpy()@s}, {-f_learning}")
    print(f"u: {u}, f_nominal: {f_nominal}, s: {s}, f_learning: {f_learning}\n")
    
    
    # Simple attitude computation for small angles
    # For more accurate control, proper attitude computation should be used
    if thrust_magnitude > 1e-6:
        # Desired acceleration in body frame
        u_normalized = u / thrust_magnitude
        
        # Convert to desired attitude (simplified)
        roll_desired = math.atan2(-u_normalized[1], -u_normalized[2])
        pitch_desired = math.atan2(u_normalized[0], math.sqrt(u_normalized[1]**2 + u_normalized[2]**2))
        # print(math.degrees(roll_desired), math.degrees(pitch_desired))
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
    # Assuming hover throttle around 0.5 and max thrust = 2*weight
    max_thrust = UAV_mass * 9.81 * 2.0
    throttle = max(0.0, min(1.0, thrust_magnitude / max_thrust * 0.5 + 0.5))

    # print(f"NeuralFly - Pos: [{current_pos[0]:.2f}, {current_pos[1]:.2f}, {current_pos[2]:.2f}] | "
    #       f"Throttle: {throttle:.3f}, Roll: {math.degrees(roll_desired):.1f}°, "
    #       f"Pitch: {math.degrees(pitch_desired):.1f}°, Yaw: {math.degrees(yaw_desired):.1f}°")

    return throttle, roll_desired, pitch_desired, yaw_desired, a_hat_new, P_new, u.tolist()

class AirSimNeuralFlyController:
    def __init__(self, phi_net_path="neural-fly_dim-a-4_v-q-pwm-epoch-199"):
        # Connect to AirSim
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        
        # Load trained phi network
        try:
            self.phi_net = load_model(phi_net_path)
            # self.phi_net.eval()
            print(f"Loaded phi network from {phi_net_path}")
            
            # Initialize adaptive parameters based on network architecture
            # Input: velocity (3) + quaternion (4) + rotor speeds (4) = 11 total
            with torch.no_grad():
                dummy_input = torch.zeros(11)  # velocity + quaternion + rotor speeds
                dummy_output = self.phi_net.phi(dummy_input)
                print("dummy_output.shape", dummy_output.shape)
                print(dummy_output)
                if dummy_output.dim() == 1:
                    h = dummy_output.shape[0] // 3  # Assume output is [3*h]
                else:
                    h = dummy_output.shape[-1]  # Assume output is [3, h] or [h, 3]
                h = 4
                print(f"Phi network output dimension h: {h}")
            # Initialize adaptive parameters
            self.a_hat = torch.zeros(h, dtype=torch.float64)
            self.P = torch.eye(h, dtype=torch.float64)
            
        except Exception as e:
            print(f"Warning: Could not load phi network from {phi_net_path}: {e}")
            print("Using dummy network - controller will not perform optimally")
            # Create a dummy network for testing
            class DummyPhiNet(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.fc = torch.nn.Linear(11, 30)  # 11 inputs: vel(3) + quat(4) + rotors(4)
                    
                def forward(self, x):
                    return self.fc(x).view(3, 10)  # Reshape to [3, 10]
            
            self.phi_net = DummyPhiNet()
            self.a_hat = torch.zeros(10, dtype=torch.float64)  # 10 features per dimension
            self.P = torch.eye(10, dtype=torch.float64)
        
        # Controller parameters
        self.dt = 0.01  # control frequency
        self.simulation_time = 0.0
        
        # History buffers for position and attitude (controller needs derivatives)
        self.pos_history = [[], [], []]  # x, y, z
        self.vel_history = [[], [], []]  # vx, vy, vz
        self.att_history = [[], [], []]  # roll, pitch, yaw
        self.ang_vel_history = [[], [], []]  # wx, wy, wz
        self.posd_history = [[], [], []] # desired position
        self.attd_history = [[], [], []] # desired attitude
        
        # Data logging
        self.time_log = []
        self.position_log = []
        self.attitude_log = []
        self.control_log = []
        self.desired_pos_log = []
        self.desired_att_log = []
        
        print("AirSim Neural-Fly Controller initialized")
        
    def get_state(self):
        """Get current drone state from AirSim"""
        # Get multirotor state
        state = self.client.getMultirotorState()
        
        # Position in NED coordinates
        pos_ned = state.kinematics_estimated.position
        position = [pos_ned.x_val, pos_ned.y_val, pos_ned.z_val]
        
        # Velocity in NED coordinates
        vel_ned = state.kinematics_estimated.linear_velocity
        velocity = [vel_ned.x_val, vel_ned.y_val, vel_ned.z_val]
        
        # Orientation quaternion (AirSim uses w,x,y,z format)
        orientation = state.kinematics_estimated.orientation
        qw = orientation.w_val
        qx = orientation.x_val
        qy = orientation.y_val
        qz = orientation.z_val
        
        # Convert to Euler angles
        roll, pitch, yaw = quaternion_to_euler(qx, qy, qz, qw)
        attitude = [roll, pitch, yaw]
        
        # Angular velocity in body frame
        ang_vel = state.kinematics_estimated.angular_velocity
        angular_velocity = [ang_vel.x_val, ang_vel.y_val, ang_vel.z_val]
        
        return position, velocity, attitude, angular_velocity
    
    
    def update_history(self, pos, vel, att, ang_vel, posd, attd):
        """Update position, velocity, attitude and angular velocity history buffers"""
        # Add current values
        for i in range(3):
            self.pos_history[i].append(pos[i])
            self.vel_history[i].append(vel[i])
            self.att_history[i].append(att[i])
            self.ang_vel_history[i].append(ang_vel[i])
            self.posd_history[i].append(posd[i])
            self.attd_history[i].append(attd[i])
        
        # Keep only last 10 elements for numerical derivatives
        max_history = 10
        for i in range(3):
            if len(self.pos_history[i]) > max_history:
                self.pos_history[i] = self.pos_history[i][-max_history:]
                self.vel_history[i] = self.vel_history[i][-max_history:]
                self.att_history[i] = self.att_history[i][-max_history:]
                self.ang_vel_history[i] = self.ang_vel_history[i][-max_history:]
                self.posd_history[i] = self.posd_history[i][-max_history:]
                self.attd_history[i] = self.attd_history[i][-max_history:]
    
    def send_control_to_airsim(self, throttle, roll_desired, pitch_desired, yaw_desired):
        """Send attitude and throttle commands to AirSim"""
        try:
            # Send roll, pitch, yaw (in radians) and throttle (0-1) to AirSim
            self.client.moveByRollPitchYawThrottleAsync(
                roll_desired, pitch_desired, yaw_desired, throttle,
                duration=self.dt
            )
        except Exception as e:
            print(f"Error sending control command: {e}")
    
    def run_simulation(self, total_time=30.0, trajectory_func=None):
        """Run the adaptive control simulation"""
        if trajectory_func is None:
            trajectory_func = test1
            
        print(f"Starting simulation for {total_time} seconds...")
        print("Taking off...")
        
        # Take off
        self.client.takeoffAsync().join()
        time.sleep(2)
        
        start_time = time.time()
        step_count = 0
        
        while self.simulation_time < total_time:
            loop_start = time.time()
            
            # Get current state
            current_pos, current_vel, current_att, current_ang_vel = self.get_state()
            
            # Get desired trajectory
            xd, yd, zd, psid = trajectory_func(self.simulation_time)
            desired_pos = [xd, yd, zd]
            desired_att = [0.0, 0.0, psid]  # Desired roll and pitch will be computed by controller
            
            # Update history buffers
            self.update_history(current_pos, current_vel, current_att, current_ang_vel, desired_pos, desired_att)
            
            # Only run controller if we have enough history
            if len(self.pos_history[0]) >= 3 and len(self.attd_history[0]) >= 3:
                
                # Run neural-fly controller
                throttle, roll_desired, pitch_desired, yaw_desired, self.a_hat, self.P, original_u = neural_fly_controller(
                    self.pos_history, self.vel_history, self.att_history, self.ang_vel_history,
                    self.posd_history, self.attd_history,
                    self.phi_net, self.a_hat, self.P, self.dt, self.simulation_time, self.client
                )
                
                # Update desired attitude with computed values
                desired_att[0] = roll_desired  # roll
                desired_att[1] = pitch_desired  # pitch
                desired_att[2] = yaw_desired   # yaw
                
                # Send control commands to AirSim
                self.send_control_to_airsim(throttle, roll_desired, pitch_desired, yaw_desired)
                
                # Log data (convert throttle back to force for consistency)
                # thrust_force = (throttle - 0.5) * 2 * UAV_mass * 9.81
                self.time_log.append(self.simulation_time)
                self.position_log.append(current_pos.copy())
                self.attitude_log.append(current_att.copy())
                self.control_log.append([throttle, roll_desired, pitch_desired, yaw_desired])
                self.desired_pos_log.append(desired_pos.copy())
                self.desired_att_log.append(desired_att.copy())
                
                # Print status every second
                if step_count % 50 == 0:
                    print(f"Time: {self.simulation_time:.1f}s")
                    print(f"  Position: [{current_pos[0]:.2f}, {current_pos[1]:.2f}, {current_pos[2]:.2f}]")
                    print(f"  Velocity: [{current_vel[0]:.2f}, {current_vel[1]:.2f}, {current_vel[2]:.2f}]")
                    print(f"  Target:   [{xd:.2f}, {yd:.2f}, {zd:.2f}]")
                    print(f"  Attitude: [{math.degrees(current_att[0]):.1f}°, {math.degrees(current_att[1]):.1f}°, {math.degrees(current_att[2]):.1f}°]")
                    print(f"original_u: {original_u}")
                    print(f"  Controls: Throttle={throttle:.3f}, Roll={math.degrees(roll_desired):.1f}°, Pitch={math.degrees(pitch_desired):.1f}°, Yaw={math.degrees(yaw_desired):.1f}°")
                    print()
            
            # Update simulation time
            step_count += 1
            self.simulation_time += self.dt
            
            # Maintain control frequency
            elapsed = time.time() - loop_start
            sleep_time = max(0, self.dt - elapsed)
            time.sleep(sleep_time)
        
        # Land and cleanup
        print("Landing...")
        self.client.landAsync().join()
        self.client.armDisarm(False)
        self.client.enableApiControl(False)
        
        print("Simulation completed!")
        
        # Print final results
        if self.position_log:
            final_pos = self.position_log[-1]
            final_target = trajectory_func(total_time)[:3]
            print(f"Final position: [{final_pos[0]:6.2f}, {final_pos[1]:6.2f}, {final_pos[2]:6.2f}]")
            print(f"Final target:   [{final_target[0]:6.2f}, {final_target[1]:6.2f}, {final_target[2]:6.2f}]")
            
            # Calculate final error
            error = np.array(final_pos) - np.array(final_target)
            print(f"Final error:    [{error[0]:6.2f}, {error[1]:6.2f}, {error[2]:6.2f}]")
            print(f"Final error norm: {np.linalg.norm(error):.2f} m")
        
        return self.get_logged_data()
    
    def get_logged_data(self):
        """Return logged data as numpy arrays"""
        return {
            'time': np.array(self.time_log),
            'position': np.array(self.position_log),
            'attitude': np.array(self.attitude_log),
            'control': np.array(self.control_log),
            'desired_position': np.array(self.desired_pos_log),
            'desired_attitude': np.array(self.desired_att_log)
        }
    
    def plot_results(self, data):
        """Plot simulation results"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Position plots
        for i, label in enumerate(['X', 'Y', 'Z']):
            axes[0, i].plot(data['time'], data['position'][:, i], 'b-', label='Actual', linewidth=2)
            axes[0, i].plot(data['time'], data['desired_position'][:, i], 'r--', label='Desired', linewidth=2)
            axes[0, i].set_xlabel('Time (s)')
            axes[0, i].set_ylabel(f'{label} Position (m)')
            axes[0, i].set_title(f'{label}-Position Tracking')
            axes[0, i].legend()
            axes[0, i].grid(True)
        
        # Attitude plots
        for i, label in enumerate(['Roll', 'Pitch', 'Yaw']):
            axes[1, i].plot(data['time'], np.degrees(data['attitude'][:, i]), 'b-', label='Actual', linewidth=2)
            axes[1, i].plot(data['time'], np.degrees(data['desired_attitude'][:, i]), 'r--', label='Desired', linewidth=2)
            axes[1, i].set_xlabel('Time (s)')
            axes[1, i].set_ylabel(f'{label} (degrees)')
            axes[1, i].set_title(f'{label} Tracking')
            axes[1, i].legend()
            axes[1, i].grid(True)
        
        plt.tight_layout()
        plt.show()
        
        # Control signals plot
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        control_labels = ['Throttle (%)', 'Roll Desired (deg)', 'Pitch Desired (deg)', 'Yaw Desired (deg)']
        
        for i in range(4):
            row, col = i // 2, i % 2
            if i == 0:
                # Thrust in Newtons
                axes[row, col].plot(data['time'], data['control'][:, i], 'g-', linewidth=2)
            else:
                # Angles in degrees
                axes[row, col].plot(data['time'], np.degrees(data['control'][:, i]), 'g-', linewidth=2)
            axes[row, col].set_xlabel('Time (s)')
            axes[row, col].set_ylabel(control_labels[i])
            axes[row, col].set_title(f'Control Signal: {control_labels[i]}')
            axes[row, col].grid(True)
        
        plt.tight_layout()
        # plt.show()


def main():
    """Main function to run the AirSim neural-fly control simulation"""
    try:
        # Create controller instance
        controller = AirSimNeuralFlyController()
        selected_traj = test1
        sim_time = 0.5
        
        # Run simulation
        data = controller.run_simulation(total_time=sim_time, trajectory_func=selected_traj)
        
        # Plot results
        if len(data['time']) > 0:
            controller.plot_results(data)
        else:
            print("No data to plot - simulation may have failed")
            
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user")
    except Exception as e:
        print(f"Error during simulation: {e}")
        import traceback
        traceback.print_exc()

def test1(t):
    return 0.0, 0.0, -5.0-t, 0.0

def test2(t):
    """Figure-8 trajectory in X-Y plane at 2 m height"""
    x_desired = 5.0 * math.sin(t * 0.5)  # Slower frequency for smoother trajectory
    y_desired = 5.0 * math.sin(t * 0.5) * math.cos(t * 0.5)
    z_desired = -10.0  # 2 meters altitude (negative in NED frame)
    yaw_desired = 0.0  # Keep yaw constant
    return x_desired, y_desired, z_desired, yaw_desired

if __name__ == "__main__":
    main()