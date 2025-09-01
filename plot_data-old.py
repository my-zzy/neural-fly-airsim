#!/usr/bin/env python
"""
Flight Data Analysis and Visualization Script

This script analyzes quadrotor flight data from CSV files in data_pinn/test/
and creates comprehensive visualizations including:
1. 2D trajectory plots (X-Y plane)
2. 3D trajectory plots
3. Heatmaps showing squared position errors

The data contains columns:
- p: actual position [x, y, z]
- p_d: desired position [x_d, y_d, z_d]
- t: time
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import ast
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LogNorm
import os

def parse_position_data(position_str):
    """Parse position string '[x, y, z]' to numpy array"""
    try:
        return np.array(ast.literal_eval(position_str))
    except:
        # Fallback parsing in case of formatting issues
        clean_str = position_str.strip('[]').replace(' ', '')
        return np.array([float(x) for x in clean_str.split(',')])

def load_flight_data(csv_file):
    """Load and process flight data from CSV file"""
    df = pd.read_csv(csv_file)
    
    # Parse position and desired position data
    positions = np.array([parse_position_data(p) for p in df['p']])
    desired_positions = np.array([parse_position_data(p) for p in df['p_d']])
    
    # Extract time data
    time = df['t'].values
    
    return {
        'time': time,
        'position': positions,
        'desired_position': desired_positions,
        'error': positions - desired_positions,
        'squared_error': np.sum((positions - desired_positions)**2, axis=1),
        'filename': Path(csv_file).stem
    }

def plot_2d_trajectories(flight_data_list, plots_dir):
    """Create 2D trajectory plot (X-Y plane) for all datasets"""
    plt.figure(figsize=(12, 10))
    
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    linestyles = ['-', '--', '-.', ':', '-']
    
    # Plot desired trajectory only once (using first dataset)
    pos_d = flight_data_list[0]['desired_position']
    plt.plot(pos_d[:, 0], pos_d[:, 1], 
            color='black', linestyle=':', linewidth=2, 
            label='Desired Trajectory', alpha=0.8, zorder=1)
    
    for i, data in enumerate(flight_data_list):
        pos = data['position']
        label = data['filename'].replace('simpleflight_fig8_', '').replace('_', ' ').title()
        
        color = colors[i % len(colors)]
        linestyle = linestyles[i % len(linestyles)]
        
        # Plot actual trajectory
        plt.plot(pos[:, 0], pos[:, 1], 
                color=color, linestyle=linestyle, linewidth=2, 
                label=f'{label}', alpha=0.8, zorder=2)
        
        # Mark start and end points
        plt.scatter(pos[0, 0], pos[0, 1], color=color, marker='o', s=100, 
                   edgecolor='black', linewidth=2, zorder=10)
        plt.scatter(pos[-1, 0], pos[-1, 1], color=color, marker='s', s=100, 
                   edgecolor='black', linewidth=2, zorder=10)
    
    plt.xlabel('X Position (m)', fontsize=12)
    plt.ylabel('Y Position (m)', fontsize=12)
    plt.title('2D Flight Trajectories Comparison (X-Y Plane)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, '2d_trajectories.png'), dpi=300, bbox_inches='tight')
    plt.show()

def plot_3d_trajectories(flight_data_list, plots_dir):
    """Create 3D trajectory plot for all datasets"""
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    linestyles = ['-', '--', '-.', ':', '-']
    
    # Plot desired trajectory only once (using first dataset)
    pos_d = flight_data_list[0]['desired_position']
    ax.plot(pos_d[:, 0], pos_d[:, 1], pos_d[:, 2], 
           color='black', linestyle=':', linewidth=2, 
           label='Desired Trajectory', alpha=0.8, zorder=1)
    
    for i, data in enumerate(flight_data_list):
        pos = data['position']
        label = data['filename'].replace('simpleflight_fig8_', '').replace('_', ' ').title()
        
        color = colors[i % len(colors)]
        linestyle = linestyles[i % len(linestyles)]
        
        # Plot actual trajectory
        ax.plot(pos[:, 0], pos[:, 1], pos[:, 2], 
               color=color, linestyle=linestyle, linewidth=2, 
               label=f'{label}', alpha=0.8, zorder=2)
        
        # Mark start and end points
        ax.scatter(pos[0, 0], pos[0, 1], pos[0, 2], color=color, marker='o', s=100, 
                  edgecolor='black', linewidth=2, zorder=10)
        ax.scatter(pos[-1, 0], pos[-1, 1], pos[-1, 2], color=color, marker='s', s=100, 
                  edgecolor='black', linewidth=2, zorder=10)
    
    ax.set_xlabel('X Position (m)', fontsize=12)
    ax.set_ylabel('Y Position (m)', fontsize=12)
    ax.set_zlabel('Z Position (m)', fontsize=12)
    ax.set_title('3D Flight Trajectories Comparison', fontsize=14, fontweight='bold')
    # ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Set equal aspect ratio
    # max_range = 0
    # for data in flight_data_list:
    #     pos = data['position']
    #     max_range = max(max_range, np.max(np.abs(pos)))
    
    # ax.set_xlim(-max_range, max_range)
    # ax.set_ylim(-max_range, max_range)
    # ax.set_zlim(-max_range, 0)  # Z is typically negative (NED frame)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, '3d_trajectories.png'), dpi=300, bbox_inches='tight')
    plt.show()

def plot_error_heatmaps(flight_data_list, plots_dir):
    """Create heatmaps showing squared position errors over time - Two rows for two methods"""
    n_datasets = len(flight_data_list)
    
    # Define the order of columns (wind conditions) - EDIT THIS ARRAY TO CHANGE ORDER
    # You can reorder, add, or remove conditions as needed
    column_order = [
        'nowind',
        '5wind', 
        '10wind',
        '12wind',
        '13p5wind',
        '15wind',
        '18sint'
    ]
    
    # Group data by method
    methods_data = {}
    for data in flight_data_list:
        filename = data['filename']
        # Extract method from filename (assuming format: simpleflight_fig8_METHOD_CONDITION)
        parts = filename.split('_')
        if len(parts) >= 3:
            method = parts[2]  # adaptive, pid, nnadaptive, etc.
            condition = '_'.join(parts[3:])  # wind condition
            
            if method not in methods_data:
                methods_data[method] = []
            methods_data[method].append((data, condition))
    
    # Sort methods for consistent ordering
    method_names = sorted(methods_data.keys())
    
    if len(method_names) < 2:
        print("Warning: Less than 2 methods found. Using original single-row layout.")
        # Fall back to original function behavior
        plot_error_heatmaps_original(flight_data_list, plots_dir)
        return
    
    # First pass: find global min/max for consistent color scaling
    all_squared_errors = []
    for data in flight_data_list:
        errors = data['error']
        squared_errors = np.sum(errors**2, axis=1)
        all_squared_errors.extend(squared_errors)
    
    vmin = np.min(all_squared_errors)
    vmax_original = np.max(all_squared_errors)
    
    # Set a threshold for color saturation (adjust this value based on your data)
    # You can experiment with different percentiles or fixed values
    error_threshold = np.percentile(all_squared_errors, 95)  # 95th percentile
    # Alternative: use a fixed threshold, e.g., error_threshold = 0.1
    
    vmax = error_threshold
    print(f"Color scale: vmin={vmin:.6f}, vmax={vmax:.6f} (threshold), original_max={vmax_original:.6f}")
    
    # Create figure with 4 rows and 7 columns (accommodate up to 4 methods)
    # Add extra height for title to prevent overlapping
    num_rows = min(4, len(method_names))
    base_height = 4 * num_rows
    title_height = 1.5  # Extra space for title
    fig, axes = plt.subplots(num_rows, 7, figsize=(24, base_height + title_height))
    
    # Handle single row case
    if num_rows == 1:
        axes = axes.reshape(1, -1)
    
    # Individual heatmaps for each method and condition
    scatters = []
    
    # Plot all available methods (up to 4)
    for row_idx, method in enumerate(method_names[:4]):
        method_datasets = methods_data[method]
        
        # Custom sorting based on column_order array
        def sort_by_condition_order(item):
            data, condition = item
            try:
                # Return the index in column_order for sorting
                return column_order.index(condition)
            except ValueError:
                # If condition not in column_order, put it at the end
                print(f'{condition} not found')
                return len(column_order)
        
        method_datasets.sort(key=sort_by_condition_order)
        print(f'Method: {method}')
        print(f'Sorted conditions: {[condition for data, condition in method_datasets]}')
        
        for col_idx in range(7):
            ax = axes[row_idx, col_idx]
            
            if col_idx < len(method_datasets):
                data, condition = method_datasets[col_idx]
                pos = data['position']
                errors = data['error']
                
                # Use X-Y position as spatial coordinates and color by squared error
                squared_errors = np.sum(errors**2, axis=1)
                
                # Create scatter plot colored by squared error with consistent scaling
                scatter = ax.scatter(pos[:, 0], pos[:, 1], c=squared_errors, 
                                   cmap='RdYlGn_r', s=8, alpha=0.8, vmin=vmin, vmax=vmax)
                scatters.append(scatter)
                
                ax.set_title(f'{method.upper()}\n{condition}', fontsize=10, fontweight='bold')
                ax.set_xlabel('X (m)', fontsize=8)
                ax.set_ylabel('Y (m)', fontsize=8)
                ax.grid(True, alpha=0.3)
                ax.axis('equal')
                ax.tick_params(axis='both', which='major', labelsize=6)
            else:
                # Hide empty subplots
                ax.set_visible(False)
    
    # Add a single colorbar for all heatmaps
    # Create a new axis for the colorbar (adjust position based on number of rows)
    if num_rows == 1:
        cbar_ax = fig.add_axes([0.92, 0.25, 0.015, 0.5])
    elif num_rows == 2:
        cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    elif num_rows == 3:
        cbar_ax = fig.add_axes([0.92, 0.10, 0.015, 0.8])
    else:  # 4 rows
        cbar_ax = fig.add_axes([0.92, 0.08, 0.015, 0.84])
    
    if scatters:
        cbar = fig.colorbar(scatters[0], cax=cbar_ax, label=f'Squared Error (m²) [clipped at {vmax:.3f}]')
        cbar.ax.tick_params(labelsize=8)
    
    # Add overall title with more space
    fig.suptitle('Squared Position Error Heatmaps: Method Comparison Across Wind Conditions', 
                 fontsize=14, fontweight='bold', y=0.97)
    
    # Adjust layout based on number of rows - with more top space for title
    if num_rows == 1:
        plt.subplots_adjust(left=0.05, right=0.90, top=0.80, bottom=0.15, wspace=0.3, hspace=0.4)
    elif num_rows == 2:
        plt.subplots_adjust(left=0.05, right=0.90, top=0.85, bottom=0.12, wspace=0.3, hspace=0.4)
    elif num_rows == 3:
        plt.subplots_adjust(left=0.05, right=0.90, top=0.88, bottom=0.08, wspace=0.3, hspace=0.4)
    else:  # 4 rows
        plt.subplots_adjust(left=0.05, right=0.90, top=0.90, bottom=0.06, wspace=0.3, hspace=0.4)
    plt.savefig(os.path.join(plots_dir, 'error_heatmaps_comparison.png'), dpi=300, bbox_inches='tight')
    plt.show()

def plot_error_heatmaps_original(flight_data_list, plots_dir):
    """Original heatmap function for fallback"""
    n_datasets = len(flight_data_list)
    
    # First pass: find global min/max for consistent color scaling
    all_squared_errors = []
    for data in flight_data_list:
        errors = data['error']
        squared_errors = np.sum(errors**2, axis=1)
        all_squared_errors.extend(squared_errors)
    
    vmin = np.min(all_squared_errors)
    vmax = np.max(all_squared_errors)
    
    # Create heatmap subplots (2x3 grid for 5 datasets + 1 empty)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    # Individual heatmaps for each dataset with consistent scaling
    scatters = []
    for i, data in enumerate(flight_data_list):
        if i >= 6:
            break
            
        pos = data['position']
        errors = data['error']
        
        # Use X-Y position as spatial coordinates and color by squared error
        squared_errors = np.sum(errors**2, axis=1)
        
        ax = axes[i]
        
        # Create scatter plot colored by squared error with consistent scaling
        scatter = ax.scatter(pos[:, 0], pos[:, 1], c=squared_errors, 
                           cmap='RdYlGn_r', s=10, alpha=0.8, vmin=vmin, vmax=vmax)
        scatters.append(scatter)
        
        label = data['filename'].replace('simpleflight_fig8_', '').replace('_', ' ').title()
        ax.set_title(f'{label}\nSquared Position Error Heatmap', fontsize=11, fontweight='bold')
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
    
    # Hide unused subplots
    for i in range(len(flight_data_list), len(axes)):
        axes[i].set_visible(False)
    
    # Add a single colorbar for all heatmaps
    # Create a new axis for the colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    if scatters:
        fig.colorbar(scatters[0], cax=cbar_ax, label='Squared Error (m²)')
    
    # Adjust layout manually instead of tight_layout to avoid warning
    plt.subplots_adjust(left=0.05, right=0.90, top=0.95, bottom=0.08, wspace=0.3, hspace=0.3)
    plt.savefig(os.path.join(plots_dir, 'error_heatmaps.png'), dpi=300, bbox_inches='tight')
    plt.show()

def plot_error_timeseries(flight_data_list, plots_dir):
    """Create a separate plot for squared position error vs time"""
    plt.figure(figsize=(12, 8))
    
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    for i, data in enumerate(flight_data_list):
        time = data['time']
        squared_errors = data['squared_error']
        label = data['filename'].replace('simpleflight_fig8_', '').replace('_', ' ').title()
        
        plt.plot(time, squared_errors, 
                color=colors[i % len(colors)], 
                linewidth=2, label=label, alpha=0.8)
    
    plt.title('Squared Position Error vs Time Comparison', fontsize=14, fontweight='bold')
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Squared Error (m²)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.yscale('log')  # Log scale for better visualization
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'error_timeseries.png'), dpi=300, bbox_inches='tight')
    plt.show()

def print_statistics(flight_data_list):
    """Print statistical summary of flight performance"""
    print("\n" + "="*80)
    print("FLIGHT PERFORMANCE STATISTICS")
    print("="*80)
    
    for data in flight_data_list:
        label = data['filename'].replace('simpleflight_fig8_', '').replace('_', ' ').title()
        errors = data['error']
        squared_errors = data['squared_error']
        
        print(f"\n{label}:")
        print(f"  Mean Squared Error (MSE):     {np.mean(squared_errors):.6f} m²")
        print(f"  Root Mean Squared Error:      {np.sqrt(np.mean(squared_errors)):.6f} m")
        print(f"  Maximum Error:                {np.sqrt(np.max(squared_errors)):.6f} m")
        print(f"  Standard Deviation:           {np.std(np.sqrt(squared_errors)):.6f} m")
        print(f"  Mean Absolute Error (X):      {np.mean(np.abs(errors[:, 0])):.6f} m")
        print(f"  Mean Absolute Error (Y):      {np.mean(np.abs(errors[:, 1])):.6f} m")
        print(f"  Mean Absolute Error (Z):      {np.mean(np.abs(errors[:, 2])):.6f} m")

def main():
    """Main function to load data and create all plots"""
    # Path to test data directory
    data_dir = Path("data_baseline/test0901")
    
    # Create plots directory if it doesn't exist
    plots_dir = "plots/0901"
    os.makedirs(plots_dir, exist_ok=True)
    
    if not data_dir.exists():
        print(f"Error: Directory {data_dir} not found!")
        return
    
    # Load all CSV files
    csv_files = list(data_dir.glob("*.csv"))
    
    if not csv_files:
        print(f"Error: No CSV files found in {data_dir}")
        return
    
    print(f"Found {len(csv_files)} CSV files:")
    for file in csv_files:
        print(f"  - {file.name}")
    
    # Load all flight data
    flight_data_list = []
    for csv_file in csv_files:
        try:
            data = load_flight_data(csv_file)
            flight_data_list.append(data)
            print(f"Loaded {csv_file.name}: {len(data['time'])} data points")
        except Exception as e:
            print(f"Error loading {csv_file.name}: {e}")
    
    if not flight_data_list:
        print("Error: No data could be loaded!")
        return
    
    print(f"\nSuccessfully loaded {len(flight_data_list)} datasets")
    print(f"Plots will be saved to: {os.path.abspath(plots_dir)}")
    
    # Create plots
    # print("\nGenerating 2D trajectory plot...")
    # plot_2d_trajectories(flight_data_list, plots_dir)
    
    # print("Generating 3D trajectory plot...")
    # plot_3d_trajectories(flight_data_list, plots_dir)
    
    print("Generating error heatmaps...")
    plot_error_heatmaps(flight_data_list, plots_dir)
    
    # print("Generating error time series plot...")
    # plot_error_timeseries(flight_data_list, plots_dir)
    
    # Print statistics
    # print_statistics(flight_data_list)
    
    print(f"\nAll plots generated and saved to {plots_dir} folder successfully!")

if __name__ == "__main__":
    main()