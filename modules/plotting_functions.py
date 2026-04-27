import numpy as np
import matplotlib
matplotlib.use("TkAgg") 
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

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