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
