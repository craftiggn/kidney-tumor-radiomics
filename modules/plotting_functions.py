import numpy as np
import matplotlib
matplotlib.use("TkAgg") 
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import matplotlib.colors as mcolors
from matplotlib.animation import FuncAnimation
from pathlib import Path
from tqdm import tqdm
from termcolor import colored
import matplotlib.animation as animation
from matplotlib.patches import Patch
import json

def browse_volume(vol: np.ndarray, axis: int = 0, cmap: str = "gray", figsize=(6, 6)) -> None:
    """
    show a 3-D array in a simple slider-driven viewer.

    Parameters
    ----------
    vol
        volume array, shape (slices, …) or any axis may be chosen
    axis
        which axis is the “slice” axis (will be moved to front)
    cmap
        colormap for imshow
    """

    vol = np.moveaxis(vol, axis, 0)
    n = vol.shape[0] - 1

    fig, ax = plt.subplots(figsize=figsize)
    plt.subplots_adjust(bottom=0.25)

    idx = 0

    vmin = np.min(vol)
    vmax = np.max(vol)

    im = ax.imshow(vol[idx],
                   cmap=cmap,
                   vmin=vmin,
                   vmax=vmax)

    ax.set_title(f"slice {idx}")
    ax.axis("off")

    # ---- Colorbar (pixel value legend) ----
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Pixel value")

    slider_ax = fig.add_axes([0.2, 0.1, 0.6, 0.03])
    slider = Slider(slider_ax, "Slice", 0, n,
                    valinit=0, valstep=1, valfmt="%0.0f")

    def _update(val):
        i = int(slider.val)
        im.set_data(vol[i])
        ax.set_title(f"slice {i}")
        fig.canvas.draw_idle()

    slider.on_changed(_update)

    plt.show()

def browse_overlay(img, mask, axis=0):

    img = np.moveaxis(img, axis, 0)
    mask = np.moveaxis(mask, axis, 0)

    n = img.shape[0] - 1
    unique_vals = np.unique(mask)

    # Remove background from legend if desired
    unique_vals = unique_vals[unique_vals != 0]

    # Create discrete colormap
    cmap = plt.cm.get_cmap("jet", int(np.max(mask) + 1))

    fig, ax = plt.subplots(figsize=(7, 7))
    plt.subplots_adjust(bottom=0.25)

    idx = 0

    im = ax.imshow(img[idx],
                   cmap="gray",
                   vmin=np.percentile(img, 1),
                   vmax=np.percentile(img, 99))

    overlay = ax.imshow(mask[idx],
                        cmap=cmap,
                        alpha=0.4,
                        vmin=0,
                        vmax=np.max(mask))

    ax.axis("off")

    # ---- Create legend ----
    legend_patches = []

    for val in unique_vals:
        color = cmap(int(val))
        patch = mpatches.Patch(color=color, label=f"Label {int(val)}")
        legend_patches.append(patch)

    ax.legend(handles=legend_patches,
              loc="upper right",
              title="Segmentation")

    # ---- Slider ----
    slider_ax = fig.add_axes([0.2, 0.1, 0.6, 0.03])
    slider = Slider(slider_ax, "Slice", 0, n,
                    valinit=0, valstep=1)

    def update(val):
        i = int(slider.val)
        im.set_data(img[i])
        overlay.set_data(mask[i])
        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()


def show_value_counts(arr):
    """
    Prints values in the ndarray as keys, and their counts as values.
    """
    values, counts = np.unique(arr, return_counts=True)
    
    for v, c in zip(values, counts):
        print(f"{v}: {c}")

def generate_compiled_data_gifs(compiled_dir, output_dir, fps=10):
    """
    Iterates through the compiled data directory and creates gifs (moving through z-axis) for each tumor
    for each tumor_id, showing the kidney/tumor mask and annotations.
    """
    in_path = Path(compiled_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # get all subdirectories (which correspond to your tumor_ids)
    tumor_dirs = [d for d in in_path.iterdir() if d.is_dir()]

    # color map
    cmap = mcolors.ListedColormap(['black', 'blue', 'red'])
    bounds = [-0.5, 0.5, 1.5, 2.5] # NOTE what is this for?
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    
    pbar = tqdm(tumor_dirs, desc="Generating GIFs")
    
    for t_dir in pbar:
        tumor_id = t_dir.name
        mask_path = t_dir / "mask.npy"
        tab_path = t_dir / "tabular.npy"
        
        # Skip if missing files
        if not mask_path.exists() or not tab_path.exists():
            continue
            
        # Load the saved tensors
        one_hot_mask = np.load(mask_path)
        tabular_data = np.load(tab_path)
        
        # 1. Reverse the one-hot encoding
        # This collapses (Z, Y, X, 3) back into (Z, Y, X) with values 0, 1, or 2
        mask_3d = np.argmax(one_hot_mask, axis=-1)
        z_slices = mask_3d.shape[0]
        
        # 2. Format the tabular array for the title
        # Since we dropped the column names earlier, this will just print the raw values
        tab_str = ", ".join([str(round(val, 2)) for val in tabular_data])
        title_text = f"ID: {tumor_id}\nFeatures: [{tab_str}]"
        
        # 3. Setup Matplotlib figure
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.axis('off')
        ax.set_title(title_text, fontsize=10, pad=15)
        
        # Initialize the plot with the very first slice
        img_plot = ax.imshow(mask_3d[0], cmap=cmap, norm=norm, animated=True)
        
        # 4. The animation update function
        def update(frame_idx):
            # Update the data in the plot for the current Z-slice
            img_plot.set_data(mask_3d[frame_idx])
            return [img_plot]
        
        # 5. Compile and save the animation
        anim = FuncAnimation(fig, update, frames=z_slices, blit=True)
        
        gif_path = out_path / f"{tumor_id}.gif"
        
        try:
            # Pillow is built into matplotlib's writers and handles GIFs natively
            anim.save(gif_path, writer='pillow', fps=fps)
        except Exception as e:
            pbar.write(f"Failed to save GIF for {tumor_id}: {e}")
            
        # Clear the figure to prevent massive RAM memory leaks across the loop
        plt.close(fig)

### DATASET VISUALISATION

def generate_tumor_video_from_npy(tumor_dir, out_dir, fps=15):
    """
    Reads a processed .npy mask and tabular file, checks metadata, 
    and generates an annotated video with a lower color legend.
    """
    tumor_path = Path(tumor_dir)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    tumor_id = tumor_path.name
    mask_file = tumor_path / "mask.npy"
    tabular_file = tumor_path / "tabular.npy"
    
    # Locate metadata file
    columns_file = tumor_path.parent / "tabular_columns.txt"

    if not mask_file.exists() or not tabular_file.exists():
        print(colored(f"ERROR: Could not find mask.npy or tabular.npy in {tumor_dir}", "yellow"))
        return
        
    if not columns_file.exists():
        print(colored(f"ERROR: Missing 'tabular_columns.txt' template in {tumor_path.parent}", "red"))
        return
    
    # Read dynamic feature layouts
    with open(columns_file, "r") as f:
        tabular_cols = [line.strip() for line in f.read().splitlines() if line.strip()]

    mask = np.load(mask_file)
    tabular = np.load(tabular_file)
    categorical_mask = np.argmax(mask, axis=-1)

    fig, ax = plt.subplots(figsize=(6, 6))
    cmap = ListedColormap(['black', 'royalblue', 'crimson'])

    im = ax.imshow(categorical_mask[0, :, :], cmap=cmap, vmin=0, vmax=2, animated=True)
    ax.axis('off')

    # Build Legend patches
    legend_elements = [
        Patch(facecolor='black', edgecolor='dimgray', label='Background'),
        Patch(facecolor='royalblue', label='Kidney'),
        Patch(facecolor='crimson', label='Tumor')
    ]
    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.0),
              ncol=3, fontsize=8, frameon=False)
    fig.subplots_adjust(bottom=0.12)

    def update_frame(z_index):
        im.set_array(categorical_mask[z_index, :, :])
        
        # Parse arrays safely using dynamic text maps
        decoded_strings = decode_labels_rpn(tabular, tabular_cols)
        
        ax.set_title(
            f"{tumor_id}: Z-Slice: {z_index}/{mask.shape[0]-1}, Shape: {mask.shape}\n"
            f"Labels: {decoded_strings}",
            fontsize=10
        )
        return [im]
    
    num_frames = mask.shape[0]
    ani = animation.FuncAnimation(
        fig, update_frame, frames=num_frames, blit=True, interval=1000/fps
    )

    video_path = out_path / f"{tumor_id}.mp4"
    # print(f"Saving video to {video_path}...")

    writer = animation.FFMpegWriter(fps=fps, metadata=dict(artist="KITS_Preprocessing"), bitrate=1800)
    ani.save(str(video_path), writer=writer)
    plt.close(fig)
    # print("Video generation complete.")

def decode_labels_rpn(label_array, label_cols):
    """
    Translates a saved binary tabular array back into raw strings using 
    the active columns tracked during preprocessing.
    """
    active_labels = []
    for value, col_name in zip(label_array, label_cols):
        if value == 1:
            # Strip prefixes like 'position_' or 'invasion_'
            clean_label = col_name.split('_')[-1]
            active_labels.append(clean_label)
            
    return tuple(active_labels)

### BATCH VISUALISATION
# contains synthetic data

def visualize_generator_batch(batch_X, batch_y, experiment_dir, out_dir, fps=15, batch_name_prefix="synthetic_batch"):
    """
    Takes a generated batch (X, y) and exports an annotated video for every item.
    Automatically handles layout and parsing using infrastructure files located at experiment_dir.
    
    batch_X: Volumetric matrix from generator of shape (Batch, Z, Y, X, 3)
    batch_y: Categorical target label array of shape (Batch, Num_Classes)
    experiment_dir: Path to the active experiment folder (e.g., .../kits_dataset_110_60_40_kidney)
    out_dir: Output directory where debug MP4 files will be generated
    """
    exp_path = Path(experiment_dir)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # Target existing root infrastructure files
    metadata_file = exp_path / "dataset_metadata.json"
    columns_file = exp_path / "tabular_columns.txt"
    
    if not metadata_file.exists() or not columns_file.exists():
        raise FileNotFoundError(
            f"Infrastructure configuration missing in {experiment_dir}. "
            f"Ensure dataset_metadata.json and tabular_columns.txt are present."
        )
        
    # Read experiment metadata for precise title tracking
    with open(metadata_file, "r") as f:
        metadata = json.load(f)
        
    bbox_strategy = metadata.get("bbox", "unknown")
    margin_val = metadata.get("margin", "None")
    
    # Read active column orders mapping to batch_y indexes
    with open(columns_file, "r") as f:
        tabular_cols = [line.strip() for line in f.read().splitlines() if line.strip()]

    batch_size = batch_X.shape[0]
    print(f"Visualizing batch of {batch_size} items for experiment variation: {exp_path.name}")

    for i in range(batch_size):
        mask = batch_X[i]
        labels = batch_y[i]  # Represents the parsed data from 'labels.npy'
        
        # Collapse one-hot channels back to categorical integer map
        categorical_mask = np.argmax(mask, axis=-1)

        fig, ax = plt.subplots(figsize=(6, 6))
        cmap = ListedColormap(['black', 'royalblue', 'crimson'])

        im = ax.imshow(categorical_mask[0, :, :], cmap=cmap, vmin=0, vmax=2, animated=True)
        ax.axis('off')

        # Unified Legend
        legend_elements = [
            Patch(facecolor='black', edgecolor='dimgray', label='Background'),
            Patch(facecolor='royalblue', label='Kidney'),
            Patch(facecolor='crimson', label='Tumor')
        ]
        ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.0),
                  ncol=3, fontsize=8, frameon=False)
        fig.subplots_adjust(bottom=0.15)  # Extended bottom cushion to fit multi-line titles securely

        def update_frame(z_index):
            im.set_array(categorical_mask[z_index, :, :])
            
            # Decode the target classifications dynamically using locked column index orders
            decoded_strings = decode_labels_rpn(labels, tabular_cols)
            
            ax.set_title(
                f"Batch Item: {i} | Z-Slice: {z_index}/{mask.shape[0]-1}\n"
                f"Strategy: {bbox_strategy} (Margin: {margin_val}) | Volume Shape: {mask.shape[:-1]}\n"
                f"Targets: {decoded_strings}",
                fontsize=9
            )
            return [im]
        
        num_frames = mask.shape[0]
        ani = animation.FuncAnimation(
            fig, update_frame, frames=num_frames, blit=True, interval=1000/fps
        )

        video_path = out_path / f"{batch_name_prefix}_item_{i}.mp4"
        writer = animation.FFMpegWriter(fps=fps, metadata=dict(artist="KITS_Generator_Debug"), bitrate=1800)
        ani.save(str(video_path), writer=writer)
        plt.close(fig)
        
        print(f"Saved: {video_path}")