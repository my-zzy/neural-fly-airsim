import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ast
import os
import glob
from matplotlib.colors import LinearSegmentedColormap

def parse_position_data(csv_file_path):
    """Parse CSV file and extract position data."""
    df = pd.read_csv(csv_file_path)
    
    actual_positions = []
    desired_positions = []
    time_data = df['t'].values
    
    for idx, row in df.iterrows():
        p_list = ast.literal_eval(row['p'])
        p_d_list = ast.literal_eval(row['p_d'])
        actual_positions.append(p_list)
        desired_positions.append(p_d_list)
    
    return np.array(actual_positions), np.array(desired_positions), time_data

def calculate_error_metrics(actual_pos, desired_pos):
    """Calculate error metrics."""
    pos_error = actual_pos - desired_pos
    error_magnitude = np.linalg.norm(pos_error, axis=1)
    
    return {
        'pos_error': pos_error,
        'error_magnitude': error_magnitude,
        'mean_error': np.mean(error_magnitude),
        'rms_error': np.sqrt(np.mean(error_magnitude**2)),
        'max_error': np.max(error_magnitude),
        'std_error': np.std(error_magnitude)
    }

def extract_wind_condition(filename):
    """Extract wind condition from filename."""
    if 'nowind' in filename:
        return 'No Wind', 0
    elif 'wind' in filename:
        # Extract number before 'wind'
        import re
        match = re.search(r'(\d+)wind[A-Za-z]*', filename)
        if match:
            wind_speed = int(match.group(1))
            return f'{match.group(0)}', wind_speed
    elif 'p20sint' in filename:
        return 'Variable Wind', 35  # Approximate for sorting
    return 'Unknown', 0

def compare_wind_effects():
    """Compare trajectory errors across different wind conditions."""
    
    # Get all CSV files
    csv_pattern = "test_wind/*adaptive*.csv"
    csv_files = glob.glob(csv_pattern)
    
    if not csv_files:
        print(f"No CSV files found matching pattern: {csv_pattern}")
        return
    
    print(f"Found {len(csv_files)} CSV files to analyze...")
    
    # Store results
    results = []
    
    for csv_file in csv_files:
        try:
            print(f"Processing: {os.path.basename(csv_file)}")
            
            # Parse data
            actual_pos, desired_pos, time_data = parse_position_data(csv_file)
            errors = calculate_error_metrics(actual_pos, desired_pos)
            
            # Extract wind condition
            wind_label, wind_speed = extract_wind_condition(os.path.basename(csv_file))
            
            # Store results
            results.append({
                'filename': os.path.basename(csv_file),
                'wind_label': wind_label,
                'wind_speed': wind_speed,
                'mean_error': errors['mean_error'],
                'rms_error': errors['rms_error'],
                'max_error': errors['max_error'],
                'std_error': errors['std_error'],
                'actual_pos': actual_pos,
                'desired_pos': desired_pos,
                'error_magnitude': errors['error_magnitude']
            })
            
        except Exception as e:
            print(f"Error processing {csv_file}: {e}")
            continue
    
    if not results:
        print("No valid results to display")
        return
    
    # Sort results by wind speed
    results.sort(key=lambda x: x['wind_speed'])
    
    # Create comprehensive comparison plot
    create_wind_comparison_plots(results)
    
    return results

def create_wind_comparison_plots(results):
    """Create comprehensive wind effect comparison plots."""
    
    fig = plt.figure(figsize=(20, 16))
    
    # Prepare data for plotting
    wind_labels = [r['wind_label'] for r in results]
    wind_speeds = [r['wind_speed'] for r in results]
    mean_errors = [r['mean_error'] for r in results]
    rms_errors = [r['rms_error'] for r in results]
    max_errors = [r['max_error'] for r in results]
    
    # 1. Error Metrics Comparison Bar Chart
    ax1 = plt.subplot(3, 3, 1)
    x_pos = np.arange(len(wind_labels))
    width = 0.25
    
    ax1.bar(x_pos - width, mean_errors, width, label='Mean Error', alpha=0.8, color='skyblue')
    ax1.bar(x_pos, rms_errors, width, label='RMS Error', alpha=0.8, color='orange')
    ax1.bar(x_pos + width, max_errors, width, label='Max Error', alpha=0.8, color='red')
    
    ax1.set_xlabel('Wind Condition')
    ax1.set_ylabel('Error (m)')
    ax1.set_title('Trajectory Error Comparison by Wind Condition')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(wind_labels, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Wind Speed vs RMS Error Scatter
    ax2 = plt.subplot(3, 3, 2)
    ax2.scatter(wind_speeds, rms_errors, s=100, alpha=0.7, c='red')
    for i, label in enumerate(wind_labels):
        ax2.annotate(label.split()[0], (wind_speeds[i], rms_errors[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    ax2.set_xlabel('Wind Speed (mph)')
    ax2.set_ylabel('RMS Error (m)')
    ax2.set_title('RMS Error vs Wind Speed')
    ax2.grid(True, alpha=0.3)
    
    # 3-5. Trajectory Heatmaps for Selected Cases
    selected_indices = [0, len(results)//2, -1] if len(results) >= 3 else list(range(len(results)))
    
    for i, idx in enumerate(selected_indices[:3]):
        ax = plt.subplot(3, 3, 3 + i)
        result = results[idx]
        
        scatter = ax.scatter(result['actual_pos'][:, 0], result['actual_pos'][:, 1],
                           c=result['error_magnitude'], cmap='Reds', s=20, alpha=0.7)
        ax.plot(result['desired_pos'][:, 0], result['desired_pos'][:, 1], 
               'b-', linewidth=2, alpha=0.8, label='Desired')
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title(f'{result["wind_label"]} - X-Y Trajectory')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax, label='Error (m)')
    
    # 6. Error Distribution Comparison
    ax6 = plt.subplot(3, 3, 6)
    for i, result in enumerate(results):
        ax6.hist(result['error_magnitude'], bins=30, alpha=0.5, 
                label=result['wind_label'], density=True)
    ax6.set_xlabel('Error Magnitude (m)')
    ax6.set_ylabel('Probability Density')
    ax6.set_title('Error Distribution by Wind Condition')
    ax6.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax6.grid(True, alpha=0.3)
    
    # 7. Box Plot of Error Distributions
    ax7 = plt.subplot(3, 3, 7)
    error_data = [result['error_magnitude'] for result in results]
    ax7.boxplot(error_data, labels=[r['wind_label'] for r in results])
    ax7.set_ylabel('Error Magnitude (m)')
    ax7.set_title('Error Distribution Box Plot')
    ax7.tick_params(axis='x', rotation=45)
    ax7.grid(True, alpha=0.3)
    
    # 8. Performance Metrics Table
    ax8 = plt.subplot(3, 3, 8)
    ax8.axis('off')
    
    # Create performance table
    table_data = []
    for result in results:
        error_mag = result['error_magnitude']
        good_tracking = np.sum(error_mag < 2.0) / len(error_mag) * 100
        poor_tracking = np.sum(error_mag > 5.0) / len(error_mag) * 100
        
        table_data.append([
            result['wind_label'][:12],  # Truncate for display
            f"{result['rms_error']:.2f}",
            f"{good_tracking:.1f}%",
            f"{poor_tracking:.1f}%"
        ])
    
    table = ax8.table(cellText=table_data,
                     colLabels=['Wind Condition', 'RMS Error (m)', '< 2m Error (%)', '> 5m Error (%)'],
                     cellLoc='center',
                     loc='center',
                     bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 2)
    ax8.set_title('Performance Summary Table', fontsize=12, fontweight='bold')
    
    # 9. Wind Speed vs Multiple Metrics
    ax9 = plt.subplot(3, 3, 9)
    ax9.plot(wind_speeds, mean_errors, 'o-', label='Mean Error', linewidth=2, markersize=8)
    ax9.plot(wind_speeds, rms_errors, 's-', label='RMS Error', linewidth=2, markersize=8)
    ax9_twin = ax9.twinx()
    ax9_twin.plot(wind_speeds, max_errors, '^-', color='red', label='Max Error', linewidth=2, markersize=8)
    
    ax9.set_xlabel('Wind Speed (mph)')
    ax9.set_ylabel('Mean/RMS Error (m)', color='blue')
    ax9_twin.set_ylabel('Max Error (m)', color='red')
    ax9.set_title('Error Metrics vs Wind Speed')
    ax9.grid(True, alpha=0.3)
    ax9.legend(loc='upper left')
    ax9_twin.legend(loc='upper right')
    
    plt.suptitle('Drone Trajectory Tracking: Wind Effect Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save the comparison plot
    output_path = 'test_wind/wind_effect_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Wind effect comparison saved to: {output_path}")
    
    plt.show()

def main():
    """Main function to run wind effect comparison."""
    print("=== Wind Effect Analysis on Drone Trajectory Tracking ===\n")
    
    results = compare_wind_effects()
    
    if results:
        print(f"\n=== SUMMARY ===")
        print(f"Analyzed {len(results)} different flight conditions:")
        
        for result in results:
            print(f"{result['wind_label']:15} | RMS Error: {result['rms_error']:.3f} m | Max Error: {result['max_error']:.3f} m")
        
        # Find best and worst performing conditions
        best_result = min(results, key=lambda x: x['rms_error'])
        worst_result = max(results, key=lambda x: x['rms_error'])
        
        print(f"\nBest Performance:  {best_result['wind_label']} (RMS: {best_result['rms_error']:.3f} m)")
        print(f"Worst Performance: {worst_result['wind_label']} (RMS: {worst_result['rms_error']:.3f} m)")
        print(f"Performance Delta: {worst_result['rms_error'] - best_result['rms_error']:.3f} m")

if __name__ == "__main__":
    main()
