#!/usr/bin/env python
import airsim
import time
import numpy as np
import math
import matplotlib.pyplot as plt
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
    g = -9.8

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
    thrust_force = (w_dot - dz_hat + g) * UAV_mass / (math.cos(phi) * math.cos(theta))
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
    accel_y_desired = max(-max_accel, min(max_accel, accel_y_desired))
    
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

    print(f"Velocities: u={u:.2f}, v={v:.2f}, w={w:.2f} | Throttle: {throttle:.3f}, Roll: {math.degrees(roll_desired):.1f}°, Pitch: {math.degrees(pitch_desired):.1f}°, Yaw: {math.degrees(yaw_desired):.1f}°")

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
            trajectory_func = traj.test1
            
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
                desired_att[1] = pitch_desired  # pitch
                desired_att[2] = yaw_desired   # yaw
                
                # Send control commands to AirSim
                self.send_control_to_airsim(throttle, roll_desired, pitch_desired, yaw_desired)
                
                # Log data (convert throttle back to force for consistency)
                thrust_force = (throttle - 0.5) * 2 * UAV_mass * 9.81
                self.time_log.append(self.simulation_time)
                self.position_log.append(current_pos.copy())
                self.attitude_log.append(current_att.copy())
                self.control_log.append([thrust_force, roll_desired, pitch_desired, yaw_desired])
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
        
        # Control signals plot
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        control_labels = ['Thrust Force (N)', 'Roll Desired (deg)', 'Pitch Desired (deg)', 'Yaw Desired (deg)']
        
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
        plt.show()


def main():
    """Main function to run the AirSim adaptive control simulation"""
    try:
        # Create controller instance
        controller = AirSimAdaptiveController()
        selected_traj = test1
        sim_time = 20.
        
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
    return -0.1, -0.*math.sin(t), -t, 0.0

if __name__ == "__main__":
    main()