import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json

def display_3d_from_json(json_path):
    # 1. Load the data
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    df = pd.DataFrame(data)

    # 2. Initialize the 3D Plot
    fig = plt.figure(figsize=(12, 10))
    ax3d = fig.add_subplot(1, 1, 1, projection='3d')
    
    # 3. Recreate the scatter plot
    # Using the same keys from your JSON: 'tumor_x', 'tumor_y', 'tumor_z'
    scatter = ax3d.scatter(df['tumor_x'], df['tumor_y'], df['tumor_z'], 
                           c=df['tumor_z'], cmap='viridis', s=60, alpha=0.8)
    
    # 4. Re-add labels
    for i, row in df.iterrows():
        # Ensure we handle the case_id logic
        short_label = str(row['case_id']).replace("case_", "") 
        ax3d.text(row['tumor_x'], row['tumor_y'], row['tumor_z'], short_label, size=8)

    # 5. Formatting
    ax3d.set_title("3D Tumor Dimensions (Restored from JSON)")
    ax3d.set_xlabel("X Size")
    ax3d.set_ylabel("Y Size")
    ax3d.set_zlabel("Z Size")
    fig.colorbar(scatter, ax=ax3d, label='Height (Z)', shrink=0.5)

    plt.show()

# Usage
display_3d_from_json("figures/3d_plot_data.json")