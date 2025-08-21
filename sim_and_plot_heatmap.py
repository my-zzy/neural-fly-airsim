import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ast
from matplotlib.colors import LinearSegmentedColormap
import os

def parse_position_data(csv_file_path):
    """
    Parse CSV file and extract position and desired position data.
    
    Args:
        csv_file_path (str): Path to the CSV file
        
    Returns:
        tuple: (actual_positions, desired_positions, time_data)
    """
    # Read CSV file
    df = pd.read_csv(csv_file_path)
    
    # Parse position data from string format to numpy arrays
    actual_positions = []
    desired_positions = []
    time_data = df['t'].values
    
    for idx, row in df.iterrows():
        # Parse actual position 'p'
        p_str = row['p']
        p_list = ast.literal_eval(p_str)
        actual_positions.append(p_list)
        
        # Parse desired position 'p_d'
        p_d_str = row['p_d']
        p_d_list = ast.literal_eval(p_d_str)
        desired_positions.append(p_d_list)
    
    actual_positions = np.array(actual_positions)
    desired_positions = np.array(desired_positions)
    
    return actual_positions, desired_positions, time_data

def calculate_trajectory_errors(actual_pos, desired_pos):
    """
    Calculate various error metrics for trajectory tracking.
    
    Args:
        actual_pos (np.array): Actual positions [N, 3]
        desired_pos (np.array): Desired positions [N, 3]
        
    Returns:
        dict: Dictionary containing different error metrics
    """
    # Calculate position errors
    pos_error = actual_pos - desired_pos
    
    # Calculate error magnitudes
    error_magnitude = np.linalg.norm(pos_error, axis=1)
    
    # Individual axis errors
    x_error = pos_error[:, 0]
    y_error = pos_error[:, 1]
    z_error = pos_error[:, 2]
    
    return {
        'pos_error': pos_error,
        'error_magnitude': error_magnitude,
        'x_error': x_error,
        'y_error': y_error,
        'z_error': z_error
    }

def create_trajectory_heatmap(actual_pos, desired_pos, errors, time_data, save_path=None):
    """
    Create comprehensive heatmap visualizations for trajectory tracking.
    
    Args:
        actual_pos (np.array): Actual positions
        desired_pos (np.array): Desired positions
        errors (dict): Error metrics dictionary
        time_data (np.array): Time data
        save_path (str): Optional path to save the plot
    """
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(20, 16))
    
    # Define custom colormap for errors (blue to red)
    colors = ['#0066CC', '#FFFFFF', '#CC0000']  # Blue -> White -> Red
    n_bins = 256
    error_cmap = LinearSegmentedColormap.from_list('error', colors, N=n_bins)
    
    # 1. 3D Trajectory with Error Color Coding
    ax1 = fig.add_subplot(3, 3, 1, projection='3d')
    scatter = ax1.scatter(actual_pos[:, 0], actual_pos[:, 1], actual_pos[:, 2], 
                         c=errors['error_magnitude'], cmap='hot', s=20, alpha=0.7)
    ax1.plot(desired_pos[:, 0], desired_pos[:, 1], desired_pos[:, 2], 
             'b--', linewidth=2, alpha=0.8, label='Desired')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('3D Trajectory with Error Magnitude')
    ax1.legend()
    plt.colorbar(scatter, ax=ax1, shrink=0.6, label='Error Magnitude (m)')
    
    # 2. X-Y Plane Heatmap
    ax2 = fig.add_subplot(3, 3, 2)
    scatter2 = ax2.scatter(actual_pos[:, 0], actual_pos[:, 1], 
                          c=errors['error_magnitude'], cmap='hot', s=30, alpha=0.8)
    ax2.plot(desired_pos[:, 0], desired_pos[:, 1], 'b--', linewidth=2, alpha=0.8, label='Desired')
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title('X-Y Plane: Error Magnitude Heatmap')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    plt.colorbar(scatter2, ax=ax2, label='Error Magnitude (m)')
    
    # 3. X-Z Plane Heatmap
    ax3 = fig.add_subplot(3, 3, 3)
    scatter3 = ax3.scatter(actual_pos[:, 0], actual_pos[:, 2], 
                          c=errors['error_magnitude'], cmap='hot', s=30, alpha=0.8)
    ax3.plot(desired_pos[:, 0], desired_pos[:, 2], 'b--', linewidth=2, alpha=0.8, label='Desired')
    ax3.set_xlabel('X (m)')
    ax3.set_ylabel('Z (m)')
    ax3.set_title('X-Z Plane: Error Magnitude Heatmap')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    plt.colorbar(scatter3, ax=ax3, label='Error Magnitude (m)')
    
    # 4. Y-Z Plane Heatmap
    ax4 = fig.add_subplot(3, 3, 4)
    scatter4 = ax4.scatter(actual_pos[:, 1], actual_pos[:, 2], 
                          c=errors['error_magnitude'], cmap='hot', s=30, alpha=0.8)
    ax4.plot(desired_pos[:, 1], desired_pos[:, 2], 'b--', linewidth=2, alpha=0.8, label='Desired')
    ax4.set_xlabel('Y (m)')
    ax4.set_ylabel('Z (m)')
    ax4.set_title('Y-Z Plane: Error Magnitude Heatmap')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    plt.colorbar(scatter4, ax=ax4, label='Error Magnitude (m)')
    
    # 5. Time vs Error Magnitude
    ax5 = fig.add_subplot(3, 3, 5)
    ax5.plot(time_data, errors['error_magnitude'], 'r-', linewidth=1.5, alpha=0.8)
    ax5.fill_between(time_data, errors['error_magnitude'], alpha=0.3, color='red')
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Error Magnitude (m)')
    ax5.set_title('Error Magnitude vs Time')
    ax5.grid(True, alpha=0.3)
    
    # 6. Individual Axis Errors vs Time
    ax6 = fig.add_subplot(3, 3, 6)
    ax6.plot(time_data, errors['x_error'], 'r-', label='X Error', alpha=0.8)
    ax6.plot(time_data, errors['y_error'], 'g-', label='Y Error', alpha=0.8)
    ax6.plot(time_data, errors['z_error'], 'b-', label='Z Error', alpha=0.8)
    ax6.set_xlabel('Time (s)')
    ax6.set_ylabel('Position Error (m)')
    ax6.set_title('Individual Axis Errors vs Time')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # 7. Error Distribution Histogram
    ax7 = fig.add_subplot(3, 3, 7)
    ax7.hist(errors['error_magnitude'], bins=50, alpha=0.7, color='orange', edgecolor='black')
    ax7.set_xlabel('Error Magnitude (m)')
    ax7.set_ylabel('Frequency')
    ax7.set_title('Error Magnitude Distribution')
    ax7.grid(True, alpha=0.3)
    
    # 8. 2D Heatmap of Error over Time and Position
    ax8 = fig.add_subplot(3, 3, 8)
    # Create a 2D grid for heatmap
    time_bins = np.linspace(time_data.min(), time_data.max(), 50)
    pos_magnitude = np.linalg.norm(actual_pos, axis=1)
    pos_bins = np.linspace(pos_magnitude.min(), pos_magnitude.max(), 50)
    
    # Create 2D histogram
    hist, time_edges, pos_edges = np.histogram2d(time_data, pos_magnitude, 
                                                 bins=[time_bins, pos_bins],
                                                 weights=errors['error_magnitude'])
    count_hist, _, _ = np.histogram2d(time_data, pos_magnitude, bins=[time_bins, pos_bins])
    
    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        avg_error = np.divide(hist, count_hist, out=np.zeros_like(hist), where=count_hist!=0)
    
    im = ax8.imshow(avg_error.T, origin='lower', aspect='auto', cmap='hot',
                    extent=[time_data.min(), time_data.max(), pos_magnitude.min(), pos_magnitude.max()])
    ax8.set_xlabel('Time (s)')
    ax8.set_ylabel('Position Magnitude (m)')
    ax8.set_title('Error Heatmap: Time vs Position')
    plt.colorbar(im, ax=ax8, label='Average Error (m)')
    
    # 9. Statistical Summary
    ax9 = fig.add_subplot(3, 3, 9)
    ax9.axis('off')
    
    # Calculate statistics
    stats_text = f"""
    TRAJECTORY ERROR STATISTICS
    
    Error Magnitude:
    • Mean: {np.mean(errors['error_magnitude']):.4f} m
    • Std:  {np.std(errors['error_magnitude']):.4f} m
    • Max:  {np.max(errors['error_magnitude']):.4f} m
    • RMS:  {np.sqrt(np.mean(errors['error_magnitude']**2)):.4f} m
    
    Individual Axis Errors (RMS):
    • X-axis: {np.sqrt(np.mean(errors['x_error']**2)):.4f} m
    • Y-axis: {np.sqrt(np.mean(errors['y_error']**2)):.4f} m
    • Z-axis: {np.sqrt(np.mean(errors['z_error']**2)):.4f} m
    
    Flight Duration: {time_data.max() - time_data.min():.2f} s
    Total Points: {len(time_data)}
    """
    
    ax9.text(0.1, 0.9, stats_text, transform=ax9.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    
    # Save plot if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Heatmap saved to: {save_path}")
    
    plt.show()
    
    return fig

def create_spatial_heatmap(actual_pos, desired_pos, errors, save_path=None):
    """
    Create a focused spatial heatmap showing error distribution in space.
    
    Args:
        actual_pos (np.array): Actual positions
        desired_pos (np.array): Desired positions  
        errors (dict): Error metrics dictionary
        save_path (str): Optional path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. X-Y Plane Error Heatmap with trajectory overlay
    ax1 = axes[0, 0]
    scatter1 = ax1.scatter(actual_pos[:, 0], actual_pos[:, 1], 
                          c=errors['error_magnitude'], cmap='Reds', 
                          s=50, alpha=0.8, edgecolors='black', linewidths=0.5)
    ax1.plot(desired_pos[:, 0], desired_pos[:, 1], 'b-', linewidth=3, 
             alpha=0.7, label='Desired Trajectory')
    ax1.set_xlabel('X Position (m)', fontsize=12)
    ax1.set_ylabel('Y Position (m)', fontsize=12)
    ax1.set_title('X-Y Plane: Trajectory Error Heatmap', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    cbar1 = plt.colorbar(scatter1, ax=ax1)
    cbar1.set_label('Error Magnitude (m)', fontsize=11)
    
    # 2. X-Z Plane Error Heatmap  
    ax2 = axes[0, 1]
    scatter2 = ax2.scatter(actual_pos[:, 0], actual_pos[:, 2], 
                          c=errors['error_magnitude'], cmap='Reds',
                          s=50, alpha=0.8, edgecolors='black', linewidths=0.5)
    ax2.plot(desired_pos[:, 0], desired_pos[:, 2], 'b-', linewidth=3,
             alpha=0.7, label='Desired Trajectory')
    ax2.set_xlabel('X Position (m)', fontsize=12)
    ax2.set_ylabel('Z Position (m)', fontsize=12)
    ax2.set_title('X-Z Plane: Trajectory Error Heatmap', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    cbar2 = plt.colorbar(scatter2, ax=ax2)
    cbar2.set_label('Error Magnitude (m)', fontsize=11)
    
    # 3. Y-Z Plane Error Heatmap
    ax3 = axes[1, 0] 
    scatter3 = ax3.scatter(actual_pos[:, 1], actual_pos[:, 2],
                          c=errors['error_magnitude'], cmap='Reds',
                          s=50, alpha=0.8, edgecolors='black', linewidths=0.5)
    ax3.plot(desired_pos[:, 1], desired_pos[:, 2], 'b-', linewidth=3,
             alpha=0.7, label='Desired Trajectory')
    ax3.set_xlabel('Y Position (m)', fontsize=12)
    ax3.set_ylabel('Z Position (m)', fontsize=12)
    ax3.set_title('Y-Z Plane: Trajectory Error Heatmap', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    cbar3 = plt.colorbar(scatter3, ax=ax3)
    cbar3.set_label('Error Magnitude (m)', fontsize=11)
    
    # 4. Error Statistics and Color Legend
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Create a detailed error analysis
    stats_text = f"""
TRAJECTORY ERROR ANALYSIS

Error Magnitude Statistics:
• Mean:     {np.mean(errors['error_magnitude']):.4f} m
• Median:   {np.median(errors['error_magnitude']):.4f} m
• Std Dev:  {np.std(errors['error_magnitude']):.4f} m
• RMS:      {np.sqrt(np.mean(errors['error_magnitude']**2)):.4f} m
• Max:      {np.max(errors['error_magnitude']):.4f} m
• 95th %ile: {np.percentile(errors['error_magnitude'], 95):.4f} m

Individual Axis RMS Errors:
• X-axis:   {np.sqrt(np.mean(errors['x_error']**2)):.4f} m
• Y-axis:   {np.sqrt(np.mean(errors['y_error']**2)):.4f} m  
• Z-axis:   {np.sqrt(np.mean(errors['z_error']**2)):.4f} m

Tracking Performance:
• Points with error < 1m:   {np.sum(errors['error_magnitude'] < 1.0)}/{len(errors['error_magnitude'])} ({100*np.sum(errors['error_magnitude'] < 1.0)/len(errors['error_magnitude']):.1f}%)
• Points with error < 5m:   {np.sum(errors['error_magnitude'] < 5.0)}/{len(errors['error_magnitude'])} ({100*np.sum(errors['error_magnitude'] < 5.0)/len(errors['error_magnitude']):.1f}%)
• Points with error > 10m:  {np.sum(errors['error_magnitude'] > 10.0)}/{len(errors['error_magnitude'])} ({100*np.sum(errors['error_magnitude'] > 10.0)/len(errors['error_magnitude']):.1f}%)

Total Trajectory Points: {len(errors['error_magnitude'])}
    """
    
    ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.8', facecolor='lightblue', alpha=0.8))
    
    plt.suptitle('Drone Trajectory Tracking Error Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save plot if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Spatial heatmap saved to: {save_path}")
    
    plt.show()
    return fig

def process_single_file(csv_file_path, create_comprehensive=True, create_spatial=True):
    """
    Process a single CSV file and create heatmaps.
    
    Args:
        csv_file_path (str): Path to CSV file
        create_comprehensive (bool): Create comprehensive analysis
        create_spatial (bool): Create focused spatial heatmap
    """
    try:
        print(f"\nProcessing: {csv_file_path}")
        print("Reading and parsing CSV data...")
        actual_pos, desired_pos, time_data = parse_position_data(csv_file_path)
        
        print("Calculating trajectory errors...")
        errors = calculate_trajectory_errors(actual_pos, desired_pos)
        
        # Print summary statistics
        print(f"\nSUMMARY STATISTICS for {os.path.basename(csv_file_path)}:")
        print(f"Mean Error: {np.mean(errors['error_magnitude']):.4f} m")
        print(f"RMS Error:  {np.sqrt(np.mean(errors['error_magnitude']**2)):.4f} m")
        print(f"Max Error:  {np.max(errors['error_magnitude']):.4f} m")
        print(f"Flight Duration: {time_data.max() - time_data.min():.2f} s")
        
        if create_comprehensive:
            print("Creating comprehensive trajectory error heatmap...")
            output_path = csv_file_path.replace('.csv', '_error_heatmap_comprehensive.png')
            create_trajectory_heatmap(actual_pos, desired_pos, errors, time_data, output_path)
        
        if create_spatial:
            print("Creating spatial error heatmap...")
            output_path = csv_file_path.replace('.csv', '_error_heatmap_spatial.png')
            create_spatial_heatmap(actual_pos, desired_pos, errors, output_path)
            
        return True
        
    except Exception as e:
        print(f"Error processing {csv_file_path}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """
    Main function to process CSV file(s) and create trajectory error heatmaps.
    """
    import sys
    
    # Check command line arguments
    if len(sys.argv) > 1:
        # Process specific file provided as argument
        csv_file_path = sys.argv[1]
        if not os.path.exists(csv_file_path):
            print(f"Error: CSV file not found at {csv_file_path}")
            return
        process_single_file(csv_file_path)
    else:
        # Default: process the no-wind case
        csv_file_path = "logs_test_fig8/simpleflight_fig8_adaptive_nowind.csv"
        
        # Check if file exists
        if not os.path.exists(csv_file_path):
            print(f"Error: CSV file not found at {csv_file_path}")
            print("Available CSV files in logs_test_fig8/:")
            if os.path.exists("logs_test_fig8/"):
                files = [f for f in os.listdir("logs_test_fig8/") if f.endswith('.csv')]
                for file in files:
                    print(f"  - {file}")
            return
        
        # Process the default file
        process_single_file(csv_file_path)
        
        # Ask user if they want to process other files
        print(f"\nWould you like to process other CSV files? Available options:")
        if os.path.exists("logs_test_fig8/"):
            csv_files = [f for f in os.listdir("logs_test_fig8/") if f.endswith('.csv')]
            for i, file in enumerate(csv_files[:10], 1):  # Show first 10 files
                print(f"  {i}. {file}")
            if len(csv_files) > 10:
                print(f"  ... and {len(csv_files) - 10} more files")
                
        print(f"\nTo process a specific file, run:")
        print(f"python sim_and_plot_heatmap.py <path_to_csv_file>")
        print(f"\nExample:")
        print(f"python sim_and_plot_heatmap.py logs_test_fig8/simpleflight_fig8_adaptive_50wind.csv")

if __name__ == "__main__":
    main()
