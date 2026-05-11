import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from modules import data_preprocessing as dataPrep

def tumor_size_analisys(base_path, json_path, output_dir="analysis_results"):
    """
    Analyzes tumor dimensions, saves plots, and returns a summary DataFrame.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(json_path, 'r') as f:
        cases = json.load(f)

    # Data container for DataFrame
    results_data = []
    expected_labels = {0.0, 1.0, 2.0}

    for case_id in cases:
        seg_path = os.path.join(base_path, case_id, "segmentation.nii.gz")
        if not os.path.exists(seg_path):
            continue

        data_seg = dataPrep.get_instance_numpy(seg_path)
        scan_z_max = data_seg.shape[0]

        # Label validation
        unique_labels = np.unique(data_seg)
        alien_labels = [l for l in unique_labels if l not in expected_labels]
        if alien_labels:
            raise ValueError(f"Data Integrity Error in {case_id}: Found {alien_labels}.")

        # Tumor Bounding Box calculation
        coords = np.where(data_seg == 2.0)
        
        if len(coords[0]) > 0:
            z_size = int(np.max(coords[0]) - np.min(coords[0]) + 1)
            x_size = int(np.max(coords[1]) - np.min(coords[1]) + 1)
            y_size = int(np.max(coords[2]) - np.min(coords[2]) + 1)
            
            results_data.append({
                "case_id": case_id,
                "scan_z_axis": scan_z_max,
                "tumor_z": z_size,
                "tumor_x": x_size,
                "tumor_y": y_size,
                "bounding_cube_volume": z_size * x_size * y_size
            })
        else:
            print(f"Notice: No tumor found in {case_id}.")

    # 1. Create DataFrame
    df = pd.DataFrame(results_data)

    # --- Visualization ---

    # FIGURE 1: Histograms (Vertical Stack 3x1)
    fig1 = plt.figure(figsize=(10, 15))
    
    # Histogram 1: Tumor Z-size
    ax1 = fig1.add_subplot(3, 1, 1)
    ax1.hist(df['tumor_z'], bins=50, color='crimson', edgecolor='black', alpha=0.7)
    ax1.set_title("Distribution of Tumor Sizes (Z-axis)")
    ax1.set_ylabel("Frequency")

    # Histogram 2: Overall Scan Depth
    ax2 = fig1.add_subplot(3, 1, 2)
    ax2.hist(df['scan_z_axis'], bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    ax2.set_title("Distribution of Scan Depths (Total Z)")
    ax2.set_ylabel("Frequency")

    # Histogram 3: Bounding Cube Volume
    ax3 = fig1.add_subplot(3, 1, 3)
    ax3.hist(df['bounding_cube_volume'], bins=100, color='gold', edgecolor='black', alpha=0.7)
    ax3.set_title("Distribution of Tumor Bounding Cube Volumes (X*Y*Z)")
    ax3.set_xlabel("Volume (voxels³)")
    ax3.set_ylabel("Frequency")

    fig1.tight_layout()
    # Save first plot to image
    fig1.savefig("figures/histograms_analysis.png")

    # FIGURE 2: 3D Scatter Plot
    fig2 = plt.figure(figsize=(12, 10))
    ax3d = fig2.add_subplot(1, 1, 1, projection='3d')
    
    scatter = ax3d.scatter(df['tumor_x'], df['tumor_y'], df['tumor_z'], 
                           c=df['tumor_z'], cmap='viridis', s=60, alpha=0.8)
    
    for i, row in df.iterrows():
        short_label = row['case_id'].replace("case_", "") 
        ax3d.text(row['tumor_x'], row['tumor_y'], row['tumor_z'], short_label, size=8)

    ax3d.set_title("3D Tumor Dimensions")
    ax3d.set_xlabel("X Size")
    ax3d.set_ylabel("Y Size")
    ax3d.set_zlabel("Z Size")
    fig2.colorbar(scatter, ax=ax3d, label='Height (Z)', shrink=0.5)

    # 2. Save 3D plot data to JSON for later visualization
    # We convert the dataframe to a dictionary for JSON serialization
    plot_data_json = df.to_dict(orient='records')
    with open("figures/3d_plot_data.json", 'w') as f:
        json.dump(plot_data_json, f, indent=4)

    plt.show()
    return df

# Example usage:
df_results = tumor_size_analisys(base_path='/home/marcel/Desktop/kits23', json_path='data/single_tumor_cases.json')

df_results.to_json('data/tumor_instance_data.json', orient='records', indent=4)