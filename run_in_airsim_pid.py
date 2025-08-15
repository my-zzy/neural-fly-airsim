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

def pid_controller(pos, vel, att, ang_vel, posd, attd, dhat, jifen, dt, t):
    
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

    # PID integral terms (reuse jifen for consistency)
    xphi, xtheta, xpsi = jifen
    g = 9.8

    # PID gains from config.py
    kp_pos = [kp1, kp2, kp3]  # Position gains
    kd_pos = [kd1, kd2, kd3]  # Velocity gains
    ki_pos = [ki1, ki2, ki3]  # Integral gains
    
    # Position errors
    ex = x - xd
    ey = y - yd
    ez = z - zd
    
    # Velocity errors (direct from AirSim)
    ev_x = u
    ev_y = v
    ev_z = w
    
    # Integral terms update
    xphi += ex * dt  # x integral
    xtheta += ey * dt  # y integral
    xpsi += ez * dt  # z integral
    
    # Limit integral windup
    max_integral = 10.0
    # print warning if exceed limit
    # if abs(xphi) > max_integral:
    #     print(f"Warning: x integral term {xphi} exceeds limit {max_integral}")
    # if abs(xtheta) > max_integral:
    #     print(f"Warning: y integral term {xtheta} exceeds limit {max_integral}")
    # if abs(xpsi) > max_integral:
    #     print(f"Warning: z integral term {xpsi} exceeds limit {max_integral}")
    # xphi = max(-max_integral, min(max_integral, xphi))
    # xtheta = max(-max_integral, min(max_integral, xtheta))
    # xpsi = max(-max_integral, min(max_integral, xpsi))

    # PID control for altitude (z-axis)
    throttle_cmd = kp_pos[2] * (-ez) + kd_pos[2] * (-ev_z) + ki_pos[2] * (-xpsi)
    # Convert to throttle (0-1 range), where 0.5 is approximately hover
    throttle = 0.5 - throttle_cmd * 0.1  # Scale factor for throttle
    throttle = max(0.0, min(1.0, throttle))

    # PID control for horizontal position - compute desired accelerations
    accel_x_desired = kp_pos[0] * (-ex) + kd_pos[0] * (-ev_x) + ki_pos[0] * (-xphi)
    accel_y_desired = kp_pos[1] * (-ey) + kd_pos[1] * (-ev_y) + ki_pos[1] * (-xtheta)

    
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

    # Return values (keeping dhat structure for compatibility)
    dhat_new = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # Not used in PID
    jifen_new = [xphi, xtheta, xpsi]

    # print(f"Velocities: u={u:.2f}, v={v:.2f}, w={w:.2f} | Throttle: {throttle:.3f}, Roll: {math.degrees(roll_desired):.1f}°, Pitch: {math.degrees(pitch_desired):.1f}°, Yaw: {math.degrees(yaw_desired):.1f}°")

    return throttle, roll_desired, pitch_desired, yaw_desired, dhat_new, jifen_new

class AirSimPIDController:
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
        
        print("AirSim PID Controller initialized")
        
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
                
                # Run PID controller
                throttle, roll_desired, pitch_desired, yaw_desired, self.dhat, self.jifen = pid_controller(
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


def main():
    """Main function to run the AirSim adaptive control simulation"""
    try:
        # Create controller instance
        controller = AirSimPIDController()
        selected_traj = test2
        sim_time = 25
        
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
    if t < 5:
        return 0, 0, -5-2*t, 0
    elif t < 10:
        return -5*(t-5), 0, -5-2*t, 0
    elif t < 15:
        return -25, -5*(t-10), -5-2*t, 0
    else:
        return -25, -25, -5-2*t, 0

def test2(t):
    """Figure-8 trajectory in X-Y plane at 2 m height"""
    x_desired = 10.0 * math.sin(t * 0.5)  # Slower frequency for smoother trajectory
    y_desired = 10.0 * math.sin(t * 0.5) * math.cos(t * 0.5)
    z_desired = -10.0-2*t  # 2 meters altitude (negative in NED frame)??
    yaw_desired = 0.0  # Keep yaw constant
    return x_desired, y_desired, z_desired, yaw_desired

if __name__ == "__main__":
    main()