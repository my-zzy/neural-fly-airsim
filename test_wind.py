#!/usr/bin/env python
import airsim
import time
import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def quaternion_to_euler(x, y, z, w):
    """Convert quaternion to Euler angles (roll, pitch, yaw)"""
    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = math.copysign(math.pi / 2, sinp)
    else:
        pitch = math.asin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw

class WindEffectTester:
    def __init__(self):
        # Connect to AirSim
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        
        # Test parameters
        self.dt = 0.02  # 50Hz control frequency
        self.test_duration = 15.0  # seconds
        
        print("Wind Effect Tester initialized")
    
    def set_wind(self, wind_speed_x=0.0, wind_speed_y=0.0, wind_speed_z=0.0):
        """Set wind speed in NED coordinates (m/s)"""
        wind = airsim.Vector3r(wind_speed_x, wind_speed_y, wind_speed_z)
        self.client.simSetWind(wind)
        print(f"Wind set to: X={wind_speed_x:.1f}, Y={wind_speed_y:.1f}, Z={wind_speed_z:.1f} m/s")
    
    def get_state(self):
        """Get current drone state"""
        state = self.client.getMultirotorState()
        
        # Position
        pos = state.kinematics_estimated.position
        position = [pos.x_val, pos.y_val, pos.z_val]
        
        # Velocity
        vel = state.kinematics_estimated.linear_velocity
        velocity = [vel.x_val, vel.y_val, vel.z_val]
        
        # Attitude
        orientation = state.kinematics_estimated.orientation
        roll, pitch, yaw = quaternion_to_euler(
            orientation.x_val, orientation.y_val, orientation.z_val, orientation.w_val
        )
        attitude = [roll, pitch, yaw]
        
        return position, velocity, attitude
    
    def run_constant_pwm_test(self, pwm_value=0.65, wind_speed_x=0.0, wind_speed_y=0.0):
        """
        Run test with constant PWM on all motors while experiencing wind
        
        Args:
            pwm_value: PWM value for all motors (0.0 to 1.0)
            wind_speed_x: Wind speed in X direction (m/s)
            wind_speed_y: Wind speed in Y direction (m/s)
        """
        print(f"\nStarting test with PWM={pwm_value:.2f}, Wind=({wind_speed_x:.1f}, {wind_speed_y:.1f}) m/s")
        
        # Set wind conditions
        self.set_wind(wind_speed_x, wind_speed_y, 0.0)
        
        # Reset drone position
        self.client.reset()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        time.sleep(1.0)
        
        # Take off to stabilize
        self.client.takeoffAsync().join()
        time.sleep(2.0)
        
        # Data logging
        time_log = []
        position_log = []
        velocity_log = []
        attitude_log = []
        
        start_time = time.time()
        simulation_time = 0.0
        
        print("Starting constant PWM ascent...")
        
        while simulation_time < self.test_duration:
            loop_start = time.time()
            
            # Apply constant PWM to all motors
            rotor_states = [pwm_value, pwm_value, pwm_value, pwm_value]
            self.client.moveByMotorPWMsAsync(
                rotor_states[0], rotor_states[1], rotor_states[2], rotor_states[3],
                duration=self.dt
            )
            
            # Log current state
            pos, vel, att = self.get_state()
            time_log.append(simulation_time)
            position_log.append(pos.copy())
            velocity_log.append(vel.copy())
            attitude_log.append(att.copy())
            
            # Print progress
            if len(time_log) % 50 == 0:  # Every 1 second
                print(f"t={simulation_time:5.2f}s | pos=({pos[0]:6.2f},{pos[1]:6.2f},{pos[2]:6.2f}) | "
                      f"vel=({vel[0]:5.2f},{vel[1]:5.2f},{vel[2]:5.2f}) | "
                      f"att=({math.degrees(att[0]):5.1f}°,{math.degrees(att[1]):5.1f}°,{math.degrees(att[2]):5.1f}°)")
            
            # Update time
            simulation_time += self.dt
            
            # Maintain control frequency
            elapsed = time.time() - loop_start
            sleep_time = max(0, self.dt - elapsed)
            time.sleep(sleep_time)
        
        # Land
        print("Test complete, landing...")
        self.client.landAsync().join()
        
        # Clear wind
        self.set_wind(0.0, 0.0, 0.0)
        
        # Return logged data
        return {
            'time': np.array(time_log),
            'position': np.array(position_log),
            'velocity': np.array(velocity_log),
            'attitude': np.array(attitude_log),
            'wind': [wind_speed_x, wind_speed_y],
            'pwm': pwm_value
        }
    
    def run_multiple_wind_tests(self, wind_conditions, pwm_value=0.65):
        """
        Run multiple tests with different wind conditions
        
        Args:
            wind_conditions: List of (wind_x, wind_y) tuples
            pwm_value: PWM value for all tests
        """
        results = []
        
        for i, (wx, wy) in enumerate(wind_conditions):
            print(f"\n{'='*60}")
            print(f"Test {i+1}/{len(wind_conditions)}: Wind ({wx:.1f}, {wy:.1f}) m/s")
            print('='*60)
            
            data = self.run_constant_pwm_test(pwm_value, wx, wy)
            results.append(data)
            
            # Wait between tests
            time.sleep(3.0)
        
        return results
    
    def plot_results(self, results):
        """Plot comparison of different wind conditions"""
        n_tests = len(results)
        colors = plt.cm.tab10(np.linspace(0, 1, n_tests))
        
        # 3D Trajectory Plot
        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot(221, projection='3d')
        
        for i, data in enumerate(results):
            wx, wy = data['wind']
            label = f"Wind ({wx:.1f}, {wy:.1f}) m/s"
            ax.plot(data['position'][:, 0], data['position'][:, 1], data['position'][:, 2],
                   color=colors[i], linewidth=2, label=label)
            
            # Mark start and end points
            ax.scatter(data['position'][0, 0], data['position'][0, 1], data['position'][0, 2],
                      color=colors[i], s=100, marker='o', alpha=0.7)
            ax.scatter(data['position'][-1, 0], data['position'][-1, 1], data['position'][-1, 2],
                      color=colors[i], s=100, marker='s', alpha=0.7)
        
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
        ax.set_zlabel('Z Position (m)')
        ax.set_title('3D Trajectory - Wind Effect Comparison')
        ax.legend()
        ax.grid(True)
        
        # 2D Trajectory Plot (Top View)
        ax2 = fig.add_subplot(222)
        for i, data in enumerate(results):
            wx, wy = data['wind']
            label = f"Wind ({wx:.1f}, {wy:.1f}) m/s"
            ax2.plot(data['position'][:, 0], data['position'][:, 1],
                    color=colors[i], linewidth=2, label=label)
            
            # Mark start and end
            ax2.scatter(data['position'][0, 0], data['position'][0, 1],
                       color=colors[i], s=100, marker='o', alpha=0.7)
            ax2.scatter(data['position'][-1, 0], data['position'][-1, 1],
                       color=colors[i], s=100, marker='s', alpha=0.7)
        
        ax2.set_xlabel('X Position (m)')
        ax2.set_ylabel('Y Position (m)')
        ax2.set_title('2D Trajectory (Top View)')
        ax2.legend()
        ax2.grid(True)
        ax2.axis('equal')
        
        # Altitude vs Time
        ax3 = fig.add_subplot(223)
        for i, data in enumerate(results):
            wx, wy = data['wind']
            label = f"Wind ({wx:.1f}, {wy:.1f}) m/s"
            ax3.plot(data['time'], -data['position'][:, 2],  # Negative Z for altitude
                    color=colors[i], linewidth=2, label=label)
        
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Altitude (m)')
        ax3.set_title('Altitude vs Time')
        ax3.legend()
        ax3.grid(True)
        
        # Horizontal Drift
        ax4 = fig.add_subplot(224)
        for i, data in enumerate(results):
            wx, wy = data['wind']
            label = f"Wind ({wx:.1f}, {wy:.1f}) m/s"
            drift = np.sqrt(data['position'][:, 0]**2 + data['position'][:, 1]**2)
            ax4.plot(data['time'], drift,
                    color=colors[i], linewidth=2, label=label)
        
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Horizontal Drift (m)')
        ax4.set_title('Horizontal Drift vs Time')
        ax4.legend()
        ax4.grid(True)
        
        plt.tight_layout()
        plt.show()
        
        # Plot velocities
        fig2, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # X velocity
        for i, data in enumerate(results):
            wx, wy = data['wind']
            label = f"Wind ({wx:.1f}, {wy:.1f}) m/s"
            axes[0,0].plot(data['time'], data['velocity'][:, 0],
                          color=colors[i], linewidth=2, label=label)
        axes[0,0].set_xlabel('Time (s)')
        axes[0,0].set_ylabel('X Velocity (m/s)')
        axes[0,0].set_title('X Velocity vs Time')
        axes[0,0].legend()
        axes[0,0].grid(True)
        
        # Y velocity  
        for i, data in enumerate(results):
            wx, wy = data['wind']
            label = f"Wind ({wx:.1f}, {wy:.1f}) m/s"
            axes[0,1].plot(data['time'], data['velocity'][:, 1],
                          color=colors[i], linewidth=2, label=label)
        axes[0,1].set_xlabel('Time (s)')
        axes[0,1].set_ylabel('Y Velocity (m/s)')
        axes[0,1].set_title('Y Velocity vs Time')
        axes[0,1].legend()
        axes[0,1].grid(True)
        
        # Z velocity
        for i, data in enumerate(results):
            wx, wy = data['wind']
            label = f"Wind ({wx:.1f}, {wy:.1f}) m/s"
            axes[1,0].plot(data['time'], data['velocity'][:, 2],
                          color=colors[i], linewidth=2, label=label)
        axes[1,0].set_xlabel('Time (s)')
        axes[1,0].set_ylabel('Z Velocity (m/s)')
        axes[1,0].set_title('Z Velocity vs Time')
        axes[1,0].legend()
        axes[1,0].grid(True)
        
        # Attitude (pitch)
        for i, data in enumerate(results):
            wx, wy = data['wind']
            label = f"Wind ({wx:.1f}, {wy:.1f}) m/s"
            axes[1,1].plot(data['time'], np.degrees(data['attitude'][:, 1]),
                          color=colors[i], linewidth=2, label=label)
        axes[1,1].set_xlabel('Time (s)')
        axes[1,1].set_ylabel('Pitch Angle (degrees)')
        axes[1,1].set_title('Pitch Angle vs Time')
        axes[1,1].legend()
        axes[1,1].grid(True)
        
        plt.tight_layout()
        plt.show()
        
        # X-Z Plane Plot (Side View) - New Window
        fig3 = plt.figure(figsize=(10, 8))
        ax_xz = fig3.add_subplot(111)
        
        for i, data in enumerate(results):
            wx, wy = data['wind']
            label = f"Wind ({wx:.1f}, {wy:.1f}) m/s"
            # Plot X position vs Altitude (negative Z)
            ax_xz.plot(data['position'][:, 0], -data['position'][:, 2],
                      color=colors[i], linewidth=3, label=label, marker='o', markersize=2, alpha=0.8)
            
            # Mark start and end points
            ax_xz.scatter(data['position'][0, 0], -data['position'][0, 2],
                         color=colors[i], s=150, marker='o', alpha=0.9, 
                         edgecolors='black', linewidth=2)
            ax_xz.scatter(data['position'][-1, 0], -data['position'][-1, 2],
                         color=colors[i], s=150, marker='s', alpha=0.9,
                         edgecolors='black', linewidth=2)
        
        ax_xz.set_xlabel('X Position (m)', fontsize=12)
        ax_xz.set_ylabel('Altitude (m)', fontsize=12)
        ax_xz.set_title('Wind Effect Comparison - Side View (X-Z Plane)\nConstant PWM Ascent with Horizontal Wind', fontsize=14, pad=20)
        ax_xz.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax_xz.grid(True, alpha=0.3)
        
        # Add annotations for better understanding
        # ax_xz.axvline(x=0, color='gray', linestyle='--', alpha=0.5, label='No X-drift line')
        # ax_xz.text(0.02, 0.98, 'Expected behavior:\n• No wind: Straight up (X=0)\n• X-wind: Drift in +X direction\n• Combined wind: Diagonal drift', 
        #            transform=ax_xz.transAxes, fontsize=10, verticalalignment='top',
        #            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
        
        plt.tight_layout()
        plt.show()
    
    def cleanup(self):
        """Clean up AirSim connection"""
        self.client.landAsync().join()
        self.client.armDisarm(False)
        self.client.enableApiControl(False)
        self.set_wind(0.0, 0.0, 0.0)  # Clear wind
        print("Cleanup completed")

def main():
    """Main function to run wind effect tests"""
    try:
        # Create tester
        tester = WindEffectTester()
        
        # Define wind conditions to test
        wind_conditions = [
            (0.0, 0.0),   # No wind (baseline)
            (5.0, 0.0),   # 5 m/s wind in X direction
            (6.0, 0.0),   # 6 m/s wind in X direction
            (7.0, 0.0),   # 7 m/s wind in X direction
            (8.0, 0.0),   # 8 m/s wind in X direction
            (9.0, 0.0),   # 9 m/s wind in X direction
            (10.0, 0.0),   # 10 m/s wind in X direction
            (11.0, 0.0),   # 11 m/s wind in X direction
            (12.0, 0.0),   # 12 m/s wind in X direction
            (13.0, 0.0),   # 13 m/s wind in X direction
            (14.0, 0.0),   # 14 m/s wind in X direction
            (15.0, 0.0),   # 15 m/s wind in X direction
        ]
        
        # PWM value for ascent (adjust based on your drone)
        pwm_value = 0.65  # Should be enough to make drone ascend
        
        print(f"Running wind effect tests with PWM = {pwm_value}")
        print(f"Wind conditions to test: {wind_conditions}")
        
        # Run tests
        results = tester.run_multiple_wind_tests(wind_conditions, pwm_value)
        
        # Plot results
        tester.plot_results(results)
        
        # Print summary statistics
        print("\n" + "="*60)
        print("WIND EFFECT SUMMARY")
        print("="*60)
        
        for i, data in enumerate(results):
            wx, wy = data['wind']
            final_pos = data['position'][-1]
            max_drift = np.max(np.sqrt(data['position'][:, 0]**2 + data['position'][:, 1]**2))
            final_altitude = -final_pos[2]  # Negative Z is altitude
            
            print(f"Wind ({wx:4.1f}, {wy:4.1f}) m/s:")
            print(f"  Final position: ({final_pos[0]:6.2f}, {final_pos[1]:6.2f}, {final_pos[2]:6.2f})")
            print(f"  Final altitude: {final_altitude:6.2f} m")
            print(f"  Max drift:      {max_drift:6.2f} m")
            print()
        
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"Error during test: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        try:
            tester.cleanup()
        except:
            pass

if __name__ == "__main__":
    main()
