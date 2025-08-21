#!/usr/bin/env python
import airsim
import time
import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from config import *

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

def adaptive_controller(pos, vel, att, ang_vel, posd, attd, dhat, jifen, dt, t):
    
    x = pos[0][-1]
    y = pos[1][-1]
    z = pos[2][-1]
    u = vel[0][-1]  # Use direct velocity from AirSim
    v = vel[1][-1]
    w = vel[2][-1]
    phi = att[0][-1]
    theta = att[1][-1]
    psi = att[2][-1]

    xd = posd[0][-1]
    yd = posd[1][-1]
    zd = posd[2][-1]
    psid = attd[2][-1]  # Only use desired yaw

    dx_hat, dy_hat, dz_hat, dphi_hat, dtheta_hat, dpsi_hat = dhat
    xphi, xtheta, xpsi = jifen
    g = 9.8

    # Calculate desired velocity derivatives (still need numerical differentiation for desired trajectory)
    xd_dot = (posd[0][-1] - posd[0][-2])/dt if len(posd[0]) >= 2 else 0.0
    yd_dot = (posd[1][-1] - posd[1][-2])/dt if len(posd[1]) >= 2 else 0.0
    zd_dot = (posd[2][-1] - posd[2][-2])/dt if len(posd[2]) >= 2 else 0.0

    xd_dot2 = ((posd[0][-1] - posd[0][-2])/dt - (posd[0][-2] - posd[0][-3])/dt)/dt if len(posd[0]) >= 3 else 0.0
    yd_dot2 = ((posd[1][-1] - posd[1][-2])/dt - (posd[1][-2] - posd[1][-3])/dt)/dt if len(posd[1]) >= 3 else 0.0
    zd_dot2 = ((posd[2][-1] - posd[2][-2])/dt - (posd[2][-2] - posd[2][-3])/dt)/dt if len(posd[2]) >= 3 else 0.0

    # Position control - adaptive altitude control
    ez = z - zd
    ew = w - zd_dot + cz*ez
    ez_dot = ew - cz*ez
    w_dot = -cw*ew - ez + zd_dot2 - cz*ez_dot
    dz_hat_dot = lamz*ew
    dz_hat += dz_hat_dot*dt
    
    # Calculate required thrust (normalized throttle for AirSim)
    thrust_force = -(w_dot - dz_hat - g) * UAV_mass / (math.cos(phi) * math.cos(theta))
    # Convert to throttle (0-1 range), where 0.5 is approximately hover
    throttle = max(0.0, min(1.0, (thrust_force / (UAV_mass * 9.81)) * 0.5 + 0.5))

    # Horizontal position control - compute desired accelerations
    ex = x - xd
    eu = u - xd_dot + cx*ex
    ex_dot = eu - cx*ex
    u_dot = -cu*eu - ex + xd_dot2 - cx*ex_dot
    dx_hat_dot = lamx*eu
    dx_hat += dx_hat_dot*dt
    accel_x_desired = u_dot - dx_hat

    ey = y - yd
    ev = v - yd_dot + cy*ey
    ey_dot = ev - cy*ey
    v_dot = -cv*ev - ey + yd_dot2 - cy*ey_dot
    dy_hat_dot = lamy*ev
    dy_hat += dy_hat_dot*dt
    accel_y_desired = v_dot - dy_hat

    '''
    Ux = (u_dot - dx_hat)*m/U1
    Uy = (v_dot - dy_hat)*m/U1
    phid_new = math.asin(Ux*math.sin(psi) - Uy*math.cos(psi))
    thetad_new = math.asin((Ux*math.cos(psi) + Uy*math.sin(psi))/math.cos(phid_new))
    
    '''

    # Convert desired accelerations to desired attitude angles
    # For small angles: roll ≈ (accel_y * cos(yaw) - accel_x * sin(yaw)) / g
    #                  pitch ≈ (accel_x * cos(yaw) + accel_y * sin(yaw)) / g
    
    # Limit acceleration commands to prevent extreme attitudes
    max_accel = 5.0  # m/s^2
    accel_x_desired = max(-max_accel, min(max_accel, accel_x_desired))
    accel_y_desired = max(-max_accel, min(max_accel, -accel_y_desired))
    
    # Calculate desired roll and pitch based on desired accelerations
    roll_desired = -(accel_y_desired * math.cos(psi) - accel_x_desired * math.sin(psi)) / 9.81
    pitch_desired = (accel_x_desired * math.cos(psi) + accel_y_desired * math.sin(psi)) / 9.81
    
    # Limit attitude angles to reasonable values (±30 degrees)
    max_angle = math.radians(30)
    roll_desired = max(-max_angle, min(max_angle, roll_desired))
    pitch_desired = max(-max_angle, min(max_angle, pitch_desired))
    
    # Use desired yaw from trajectory
    yaw_desired = psid

    # Update disturbance estimates
    dhat_new = [dx_hat, dy_hat, dz_hat, dphi_hat, dtheta_hat, dpsi_hat]
    jifen_new = [xphi, xtheta, xpsi]

    # print(f"Velocities: u={u:.2f}, v={v:.2f}, w={w:.2f} | Throttle: {throttle:.3f}, Roll: {math.degrees(roll_desired):.1f}°, Pitch: {math.degrees(pitch_desired):.1f}°, Yaw: {math.degrees(yaw_desired):.1f}°")

    return throttle, roll_desired, pitch_desired, yaw_desired, dhat_new, jifen_new

class AirSimAdaptiveController:
    def __init__(self):
        # Connect to AirSim
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        
        # Controller parameters
        self.dt = 0.01  # control frequency
        self.simulation_time = 0.0
        
        # Initialize adaptive controller variables
        self.dhat = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # disturbance estimates
        self.jifen = [0.0, 0.0, 0.0]  # integral terms
        
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
        
        print("AirSim Adaptive Controller initialized")
        
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
                
                # Run adaptive controller
                throttle, roll_desired, pitch_desired, yaw_desired, self.dhat, self.jifen = adaptive_controller(
                    self.pos_history, self.vel_history, self.att_history, self.ang_vel_history,
                    self.posd_history, self.attd_history,
                    self.dhat, self.jifen, self.dt, self.simulation_time
                )
                
                # Update desired attitude with computed values
                desired_att[0] = roll_desired  # roll
                desired_att[1] = -pitch_desired  # pitch
                desired_att[2] = yaw_desired   # yaw
                
                # Send control commands to AirSim
                self.send_control_to_airsim(throttle, roll_desired, pitch_desired, yaw_desired)
                
                # Log data (convert throttle back to force for consistency)
                thrust_force = (throttle - 0.5) * 2 * UAV_mass * 9.81
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
    
    def calculate_trajectory_mse(self, data):
        """Calculate MSE loss between actual and desired trajectories"""
        if len(data['time']) == 0:
            print("No data available for MSE calculation")
            return None
        
        # Calculate position MSE
        pos_errors = data['position'] - data['desired_position']
        pos_mse_x = np.mean(pos_errors[:, 0]**2)
        pos_mse_y = np.mean(pos_errors[:, 1]**2)
        pos_mse_z = np.mean(pos_errors[:, 2]**2)
        pos_mse_total = np.mean(np.sum(pos_errors**2, axis=1))
        
        # Calculate attitude MSE (convert to degrees for better interpretation)
        att_errors_deg = np.degrees(data['attitude'] - data['desired_attitude'])
        att_mse_roll = np.mean(att_errors_deg[:, 0]**2)
        att_mse_pitch = np.mean(att_errors_deg[:, 1]**2)
        att_mse_yaw = np.mean(att_errors_deg[:, 2]**2)
        att_mse_total = np.mean(np.sum(att_errors_deg**2, axis=1))
        
        # Calculate RMS errors (square root of MSE)
        pos_rms_x = np.sqrt(pos_mse_x)
        pos_rms_y = np.sqrt(pos_mse_y)
        pos_rms_z = np.sqrt(pos_mse_z)
        pos_rms_total = np.sqrt(pos_mse_total)
        
        att_rms_roll = np.sqrt(att_mse_roll)
        att_rms_pitch = np.sqrt(att_mse_pitch)
        att_rms_yaw = np.sqrt(att_mse_yaw)
        att_rms_total = np.sqrt(att_mse_total)
        
        # Print results
        print("\n" + "="*60)
        print("TRAJECTORY TRACKING PERFORMANCE METRICS")
        print("="*60)
        
        print("\nPosition Tracking Errors:")
        print(f"  X-axis  - MSE: {pos_mse_x:8.4f} m²,  RMS: {pos_rms_x:8.4f} m")
        print(f"  Y-axis  - MSE: {pos_mse_y:8.4f} m²,  RMS: {pos_rms_y:8.4f} m")
        print(f"  Z-axis  - MSE: {pos_mse_z:8.4f} m²,  RMS: {pos_rms_z:8.4f} m")
        print(f"  Total   - MSE: {pos_mse_total:8.4f} m²,  RMS: {pos_rms_total:8.4f} m")
        
        print("\nAttitude Tracking Errors:")
        print(f"  Roll    - MSE: {att_mse_roll:8.4f} deg², RMS: {att_rms_roll:8.4f} deg")
        print(f"  Pitch   - MSE: {att_mse_pitch:8.4f} deg², RMS: {att_rms_pitch:8.4f} deg")
        print(f"  Yaw     - MSE: {att_mse_yaw:8.4f} deg², RMS: {att_rms_yaw:8.4f} deg")
        print(f"  Total   - MSE: {att_mse_total:8.4f} deg², RMS: {att_rms_total:8.4f} deg")
        
        # Calculate maximum errors
        pos_max_errors = np.max(np.abs(pos_errors), axis=0)
        att_max_errors_deg = np.max(np.abs(att_errors_deg), axis=0)
        
        print("\nMaximum Absolute Errors:")
        print(f"  Position - X: {pos_max_errors[0]:6.4f} m, Y: {pos_max_errors[1]:6.4f} m, Z: {pos_max_errors[2]:6.4f} m")
        print(f"  Attitude - Roll: {att_max_errors_deg[0]:6.4f}°, Pitch: {att_max_errors_deg[1]:6.4f}°, Yaw: {att_max_errors_deg[2]:6.4f}°")
        
        print("="*60)
        
        # Return metrics as dictionary
        return {
            'position': {
                'mse': {'x': pos_mse_x, 'y': pos_mse_y, 'z': pos_mse_z, 'total': pos_mse_total},
                'rms': {'x': pos_rms_x, 'y': pos_rms_y, 'z': pos_rms_z, 'total': pos_rms_total},
                'max_error': {'x': pos_max_errors[0], 'y': pos_max_errors[1], 'z': pos_max_errors[2]}
            },
            'attitude': {
                'mse': {'roll': att_mse_roll, 'pitch': att_mse_pitch, 'yaw': att_mse_yaw, 'total': att_mse_total},
                'rms': {'roll': att_rms_roll, 'pitch': att_rms_pitch, 'yaw': att_rms_yaw, 'total': att_rms_total},
                'max_error': {'roll': att_max_errors_deg[0], 'pitch': att_max_errors_deg[1], 'yaw': att_max_errors_deg[2]}
            }
        }
    
    def save_mse_metrics(self, metrics, filename="trajectory_mse_metrics.txt"):
        """Save MSE metrics to a text file"""
        if metrics is None:
            print("No metrics to save")
            return
            
        with open(filename, 'w') as f:
            f.write("TRAJECTORY TRACKING PERFORMANCE METRICS\n")
            f.write("="*60 + "\n\n")
            
            f.write("Position Tracking Errors:\n")
            f.write(f"  X-axis  - MSE: {metrics['position']['mse']['x']:8.4f} m²,  RMS: {metrics['position']['rms']['x']:8.4f} m\n")
            f.write(f"  Y-axis  - MSE: {metrics['position']['mse']['y']:8.4f} m²,  RMS: {metrics['position']['rms']['y']:8.4f} m\n")
            f.write(f"  Z-axis  - MSE: {metrics['position']['mse']['z']:8.4f} m²,  RMS: {metrics['position']['rms']['z']:8.4f} m\n")
            f.write(f"  Total   - MSE: {metrics['position']['mse']['total']:8.4f} m²,  RMS: {metrics['position']['rms']['total']:8.4f} m\n\n")
            
            f.write("Attitude Tracking Errors:\n")
            f.write(f"  Roll    - MSE: {metrics['attitude']['mse']['roll']:8.4f} deg², RMS: {metrics['attitude']['rms']['roll']:8.4f} deg\n")
            f.write(f"  Pitch   - MSE: {metrics['attitude']['mse']['pitch']:8.4f} deg², RMS: {metrics['attitude']['rms']['pitch']:8.4f} deg\n")
            f.write(f"  Yaw     - MSE: {metrics['attitude']['mse']['yaw']:8.4f} deg², RMS: {metrics['attitude']['rms']['yaw']:8.4f} deg\n")
            f.write(f"  Total   - MSE: {metrics['attitude']['mse']['total']:8.4f} deg², RMS: {metrics['attitude']['rms']['total']:8.4f} deg\n\n")
            
            f.write("Maximum Absolute Errors:\n")
            f.write(f"  Position - X: {metrics['position']['max_error']['x']:6.4f} m, Y: {metrics['position']['max_error']['y']:6.4f} m, Z: {metrics['position']['max_error']['z']:6.4f} m\n")
            f.write(f"  Attitude - Roll: {metrics['attitude']['max_error']['roll']:6.4f}°, Pitch: {metrics['attitude']['max_error']['pitch']:6.4f}°, Yaw: {metrics['attitude']['max_error']['yaw']:6.4f}°\n")
            f.write("="*60 + "\n")
            
        print(f"MSE metrics saved to {filename}")
    
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
        
        # 3D trajectory plot
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot actual trajectory
        ax.plot(data['position'][:, 0], data['position'][:, 1], data['position'][:, 2], 
                'b-', label='Actual Trajectory', linewidth=2)
        
        # Plot desired trajectory
        ax.plot(data['desired_position'][:, 0], data['desired_position'][:, 1], data['desired_position'][:, 2], 
                'r--', label='Desired Trajectory', linewidth=2)
        
        # Mark start and end points
        ax.scatter(data['position'][0, 0], data['position'][0, 1], data['position'][0, 2], 
                  color='green', s=100, label='Start', marker='o')
        ax.scatter(data['position'][-1, 0], data['position'][-1, 1], data['position'][-1, 2], 
                  color='red', s=100, label='End', marker='s')
        
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
        ax.set_zlabel('Z Position (m)')
        ax.set_title('3D Trajectory Tracking')
        ax.legend()
        ax.grid(True)
        
        # Set equal aspect ratio for better visualization
        max_range = np.array([data['position'][:, 0].max() - data['position'][:, 0].min(),
                             data['position'][:, 1].max() - data['position'][:, 1].min(),
                             data['position'][:, 2].max() - data['position'][:, 2].min()]).max()
        
        # Adjust the aspect ratio
        mid_x = (data['position'][:, 0].max() + data['position'][:, 0].min()) * 0.5
        mid_y = (data['position'][:, 1].max() + data['position'][:, 1].min()) * 0.5
        mid_z = (data['position'][:, 2].max() + data['position'][:, 2].min()) * 0.5
        
        ax.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
        ax.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
        ax.set_zlim(mid_z - max_range/2, mid_z + max_range/2)
        
        plt.tight_layout()
        plt.show()
                
        # 2D trajectory plot (X-Y)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(data['position'][:, 0], data['position'][:, 1], 'b-', label='Actual', linewidth=2)
        ax.plot(data['desired_position'][:, 0], data['desired_position'][:, 1], 'r--', label='Desired', linewidth=2)
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
        ax.set_title('2D Trajectory (X-Y)')
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        plt.show()
        
        # Control signals plot
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        control_labels = ['throttle (%)', 'Roll Desired (deg)', 'Pitch Desired (deg)', 'Yaw Desired (deg)']
        
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
    
    def plot_tracking_errors(self, data):
        """Plot tracking errors over time"""
        if len(data['time']) == 0:
            print("No data available for error plotting")
            return
            
        # Calculate errors
        pos_errors = data['position'] - data['desired_position']
        att_errors_deg = np.degrees(data['attitude'] - data['desired_attitude'])
        
        # Plot position errors
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Position error plots
        for i, label in enumerate(['X', 'Y', 'Z']):
            axes[0, i].plot(data['time'], pos_errors[:, i], 'r-', linewidth=2)
            axes[0, i].set_xlabel('Time (s)')
            axes[0, i].set_ylabel(f'{label} Error (m)')
            axes[0, i].set_title(f'{label}-Position Error')
            axes[0, i].grid(True)
            axes[0, i].axhline(y=0, color='k', linestyle='--', alpha=0.3)
        
        # Attitude error plots
        for i, label in enumerate(['Roll', 'Pitch', 'Yaw']):
            axes[1, i].plot(data['time'], att_errors_deg[:, i], 'b-', linewidth=2)
            axes[1, i].set_xlabel('Time (s)')
            axes[1, i].set_ylabel(f'{label} Error (degrees)')
            axes[1, i].set_title(f'{label} Error')
            axes[1, i].grid(True)
            axes[1, i].axhline(y=0, color='k', linestyle='--', alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Plot error magnitude over time
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Position error magnitude
        pos_error_mag = np.sqrt(np.sum(pos_errors**2, axis=1))
        ax1.plot(data['time'], pos_error_mag, 'r-', linewidth=2)
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Position Error Magnitude (m)')
        ax1.set_title('Position Tracking Error Magnitude')
        ax1.grid(True)
        
        # Attitude error magnitude
        att_error_mag = np.sqrt(np.sum(att_errors_deg**2, axis=1))
        ax2.plot(data['time'], att_error_mag, 'b-', linewidth=2)
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Attitude Error Magnitude (degrees)')
        ax2.set_title('Attitude Tracking Error Magnitude')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()


def main():
    """Main function to run the AirSim adaptive control simulation"""
    try:
        # Create controller instance
        controller = AirSimAdaptiveController()
        selected_traj = test2
        sim_time = 20.
        
        # Run simulation
        data = controller.run_simulation(total_time=sim_time, trajectory_func=selected_traj)
        
        # Plot results
        if len(data['time']) > 0:
            controller.plot_results(data)
            
            # Calculate and display MSE metrics
            mse_metrics = controller.calculate_trajectory_mse(data)
            
            # Save MSE metrics to file
            # if mse_metrics is not None:
            #     controller.save_mse_metrics(mse_metrics, f"adaptive_controller_mse_metrics_{selected_traj.__name__}.txt")
            
            # Plot tracking errors
            controller.plot_tracking_errors(data)
        else:
            print("No data to plot - simulation may have failed")
            
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user")
    except Exception as e:
        print(f"Error during simulation: {e}")
        import traceback
        traceback.print_exc()


def test1(t):
    return 0.1*t, -0.0, -5.0-1.0*t, 0.0

def test2(t):
    """Figure-8 trajectory in X-Y plane at 2 m height"""
    x_desired = 10.0 * math.sin(t * 0.5)  # Slower frequency for smoother trajectory
    y_desired = 10.0 * math.sin(t * 0.5) * math.cos(t * 0.5)
    z_desired = -10.0-2*t  # 2 meters altitude (negative in NED frame)??
    yaw_desired = 0.0  # Keep yaw constant
    return x_desired, y_desired, z_desired, yaw_desired

if __name__ == "__main__":
    main()