import numpy as np
import matplotlib
matplotlib.use("Agg") # Agg, because TkAgg was causing issues on the HPC
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
import pandas as pd
import re
import seaborn
import visualkeras
import keras

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

def plot_threshold(df, threshold, output_path='./figures/threshold.png'):
    sorted_diameters = np.sort(df['calculated_diameter_mm'].dropna())
    y_vals = np.arange(1, len(sorted_diameters) + 1)
    
    # Plot CDF
    plt.figure(figsize=(10, 4))
    plt.step(sorted_diameters, y_vals, color='teal', where='post', linewidth=2)
    plt.axvline(x=threshold, color='red', linestyle='--', label=f'Cutoff: {threshold}mm')
    
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Filter dataset
    kept_df = df[df['calculated_diameter_mm'] <= threshold].copy()
    
    plt.title(f"Tumor Size CDF (Keeping {len(kept_df)} / {len(df)} cases)")
    plt.xlabel("Calculated Tumor Diameter (mm)")
    plt.ylabel("Cumulative Case Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    

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

### MODEL FIGURES
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, confusion_matrix

def plot_and_evaluate(fold_out_dir, fold_no, class_names):
    """
    Generates an 8-subplot grid for a single fold, showing Loss, AUC, AP, and F1 curves 
    for both training and validation across each target label.
    """
    fold_out_dir = Path(fold_out_dir)
    csv_path = fold_out_dir / f"training_log_fold_{fold_no}.csv"
    
    if not csv_path.exists():
        print(f"[Warning] No log file found at {csv_path}")
        return

    df = pd.read_csv(csv_path)
    epochs = df['epoch'] + 1

    fig, axes = plt.subplots(4, 2, figsize=(12, 20)) 

    # Unpack row by row: Loss, AUC, AP, F1
    (ax_tr_loss, ax_val_loss), (ax_tr_auc, ax_val_auc), (ax_tr_ap, ax_val_ap), (ax_tr_f1, ax_val_f1) = axes

    # Setup Titles and Scales
    ax_tr_loss.set_title(f"Train Loss (Fold {fold_no})"); ax_tr_loss.set_yscale('log')
    ax_val_loss.set_title(f"Val Loss (Fold {fold_no})"); ax_val_loss.set_yscale('log')
    
    ax_tr_auc.set_title(f"Train AUC (Fold {fold_no})"); ax_tr_auc.set_ylim([0.48, 1.02])
    ax_val_auc.set_title(f"Val AUC (Fold {fold_no})"); ax_val_auc.set_ylim([0.48, 1.02])
    
    ax_tr_ap.set_title(f"Train AP (Fold {fold_no})"); ax_tr_ap.set_ylim([-0.02, 1.02])
    ax_val_ap.set_title(f"Val AP (Fold {fold_no})"); ax_val_ap.set_ylim([-0.02, 1.02])
    
    ax_tr_f1.set_title(f"Train F1 (Fold {fold_no})"); ax_tr_f1.set_ylim([-0.02, 1.02])
    ax_val_f1.set_title(f"Val F1 (Fold {fold_no})"); ax_val_f1.set_ylim([-0.02, 1.02])

    for c in class_names:
        if f'train_loss_{c}' in df: ax_tr_loss.plot(epochs, df[f'train_loss_{c}'], label=c, lw=1.5)
        if f'val_loss_{c}' in df: ax_val_loss.plot(epochs, df[f'val_loss_{c}'], label=c, lw=1.5)
        if f'train_auc_{c}' in df: ax_tr_auc.plot(epochs, df[f'train_auc_{c}'], label=c, lw=1.5)
        if f'val_auc_{c}' in df: ax_val_auc.plot(epochs, df[f'val_auc_{c}'], label=c, lw=1.5)
        if f'train_ap_{c}' in df: ax_tr_ap.plot(epochs, df[f'train_ap_{c}'], label=c, lw=1.5)
        if f'val_ap_{c}' in df: ax_val_ap.plot(epochs, df[f'val_ap_{c}'], label=c, lw=1.5)
        if f'train_f1_{c}' in df: ax_tr_f1.plot(epochs, df[f'train_f1_{c}'], label=c, lw=1.5)
        if f'val_f1_{c}' in df: ax_val_f1.plot(epochs, df[f'val_f1_{c}'], label=c, lw=1.5)

    for ax in axes.flat:
        ax.set_xlabel("Epochs")
        ax.legend(loc='best', fontsize='small', frameon=True)
        ax.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    save_path = fold_out_dir / f"training_curves_fold_{fold_no}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Success] 8-panel fold training curves saved to: {save_path}")


def plot_and_evaluate_all_folds(out_dir):
    """
    Generates an 8-subplot grid across all folds, plotting average Loss, AUC, AP, and F1 curves 
    for both training and validation (where each line represents a single Fold).
    """
    out_dir = Path(out_dir)
    fold_dirs = sorted(
        [d for d in out_dir.iterdir() if d.is_dir() and re.match(r'^fold_\d+$', d.name)],
        key=lambda x: int(re.search(r'\d+', x.name).group())
    )

    if not fold_dirs:
        return

    fig, axes = plt.subplots(2, 4, figsize=(24, 10))
    (ax_tr_loss, ax_val_loss, ax_tr_auc, ax_val_auc), (ax_tr_ap, ax_val_ap, ax_tr_f1, ax_val_f1) = axes

    ax_tr_loss.set_title("Training Loss per Fold"); ax_tr_loss.set_yscale('log')
    ax_val_loss.set_title("Validation Loss per Fold"); ax_val_loss.set_yscale('log')
    
    ax_tr_auc.set_title("Training AUC per Fold"); ax_tr_auc.set_ylim([0.48, 1.02])
    ax_val_auc.set_title("Validation AUC per Fold"); ax_val_auc.set_ylim([0.48, 1.02])
    
    ax_tr_ap.set_title("Training AP per Fold"); ax_tr_ap.set_ylim([-0.02, 1.02])
    ax_val_ap.set_title("Validation AP per Fold"); ax_val_ap.set_ylim([-0.02, 1.02])
    
    ax_tr_f1.set_title("Training F1 per Fold"); ax_tr_f1.set_ylim([-0.02, 1.02])
    ax_val_f1.set_title("Validation F1 per Fold"); ax_val_f1.set_ylim([-0.02, 1.02])

    for fold_dir in fold_dirs:
        fold_idx = re.search(r'\d+', fold_dir.name).group()
        csv_path = fold_dir / f"training_log_fold_{fold_idx}.csv"
        if not csv_path.exists(): continue

        df = pd.read_csv(csv_path)
        epochs = df['epoch'] + 1 
        label = f"Fold {int(fold_idx) + 1}"

        if 'loss' in df: ax_tr_loss.plot(epochs, df['loss'], lw=2, label=label)
        if 'val_loss' in df: ax_val_loss.plot(epochs, df['val_loss'], lw=2, label=label)
        if 'auc' in df: ax_tr_auc.plot(epochs, df['auc'], lw=2, label=label)
        if 'val_auc' in df: ax_val_auc.plot(epochs, df['val_auc'], lw=2, label=label)
        if 'ap' in df: ax_tr_ap.plot(epochs, df['ap'], lw=2, label=label)
        if 'val_ap' in df: ax_val_ap.plot(epochs, df['val_ap'], lw=2, label=label)
        if 'f1' in df: ax_tr_f1.plot(epochs, df['f1'], lw=2, label=label)
        if 'val_f1' in df: ax_val_f1.plot(epochs, df['val_f1'], lw=2, label=label)

    for ax in axes.flat:
        ax.set_xlabel("Epochs")
        ax.legend(loc='best', frameon=True)
        ax.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    save_path = out_dir / "all_folds_performance_matrix.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Success] 8-panel overall performance matrix saved to: {save_path}")


def plot_overall_confusion_matrix(all_y_true, all_y_pred, tabula_cols, out_dir):
    """
    Generates a cumulative overall confusion matrix by combining predictions across all folds.
    """
    y_true_concat = np.concatenate(all_y_true, axis=0)
    y_pred_concat = np.concatenate(all_y_pred, axis=0)

    plot_feature_confusion_matrix(
        y_true=y_true_concat,
        y_pred=y_pred_concat,
        tabula_cols=tabula_cols,
        fold_no="OVERALL",
        out_dir=out_dir
    )

def plot_multilabel_roc_and_pr_curves(y_true, y_pred, fold_no, out_dir, class_names=None):
    """
    Plots per-class ROC and Precision-Recall curves with standardized axis limits.
    """
    n_classes = y_true.shape[1]
    if class_names is None:
        class_names = [f"Class {i}" for i in range(n_classes)]

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    ax_roc, ax_pr = axes

    # --- ROC Plot (Y-axis fixed from 0.5 to 1.02) ---
    ax_roc.set_title(f"ROC Curve (Fold {fold_no})", fontsize=14, fontweight='bold', pad=10)
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_xlim([-0.02, 1.02])
    ax_roc.set_ylim([-0.02, 1.02])
    ax_roc.plot([0, 1], [0, 1], 'k--', lw=2, label="Random Chance")

    # --- PR Plot (Y-axis fixed from 0.0 to 1.02) ---
    ax_pr.set_title(f"Precision-Recall Curve (Fold {fold_no})", fontsize=14, fontweight='bold', pad=10)
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_xlim([-0.02, 1.02])
    ax_pr.set_ylim([-0.02, 1.02])

    # Generate curves for each class
    for i in range(n_classes):
        # ROC
        fpr, tpr, _ = roc_curve(y_true[:, i], y_pred[:, i])
        roc_auc = auc(fpr, tpr)
        ax_roc.plot(fpr, tpr, lw=2, label=f"{class_names[i]} (AUC = {roc_auc:.3f})")

        # Precision-Recall
        precision, recall, _ = precision_recall_curve(y_true[:, i], y_pred[:, i])
        pr_auc = average_precision_score(y_true[:, i], y_pred[:, i])
        ax_pr.plot(recall, precision, lw=2, label=f"{class_names[i]} (AP = {pr_auc:.3f})")

    ax_roc.legend(loc='lower right', frameon=True)
    ax_roc.grid(True, linestyle='--', alpha=0.6)

    ax_pr.legend(loc='lower left', frameon=True)
    ax_pr.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    save_path = Path(out_dir) / f"roc_pr_curves_fold_{fold_no}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[Success] ROC and PR curves saved to: {save_path}")

def plot_feature_confusion_matrix(y_true, y_pred, tabula_cols, fold_no, out_dir):
    """
    Splits the 6 node output into position (0,1,2) and invasion (3,4,5).
    Calculates the argmax for each group to generate two distinct 3x3 confusion matrices.     
    """

    if not tabula_cols:
        print(colored("[Skip] No tabular cols provided. Cannot plot confusion matrix.", "yellow"))
        return

    # parse and group columns by prefix
    features = {}
    for i, col_name in enumerate(tabula_cols):
        feat_name, class_name = col_name.split('_', 1)

        if feat_name not in features:
            features[feat_name] = {'indices': [], 'labels': []}

        features[feat_name]['indices'].append(i)
        features[feat_name]['labels'].append(class_name)

    # setup the figure
    n_features = len(features)
    fig, axes = plt.subplots(1, n_features, figsize=(7 * n_features, 6))

    # ensure axes is iterable
    if n_features == 1:
        axes = [axes]

    # generate a matrix for each discrete feature
    for ax, (feat_name, feat_data) in zip(axes, features.items()):
        indices = feat_data['indices']
        labels = feat_data['labels']

        # slice the true and pred arrays for just these feature indices
        y_true_sub = y_true[:, indices]
        y_pred_sub = y_pred[:, indices]

        # convert probablies to discrete class indices via argmax
        y_true_cls = np.argmax(y_true_sub, axis=1)
        y_pred_cls = np.argmax(y_pred_sub, axis=1)

        # compute and plot
        cm = confusion_matrix(y_true_cls, y_pred_cls)
        seaborn.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                        xticklabels=labels, yticklabels=labels,
                        cbar_kws={'label': 'Sample Count'})

        ax.set_title(f"{feat_name.capitalize()} Confusion Matrix (Fold {fold_no})")
        ax.set_ylabel(f"True Label")
        ax.set_xlabel(f"Predicted Label")

    plt.tight_layout()
    save_path = Path(out_dir) / f"confusion_matrix_fold_{fold_no}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"[Success] Dynamic confusion matrices saved to: {save_path}")

def plot_model_architecture(model, output_path):
    """
    Saves a 3D visual diagram of the model architecture.
    """
    output_path = Path(output_path)
    print(colored(f"Saving model architecture visual diagram to {output_path}...", "green"))

    for layer in model.layers:
        if not hasattr(layer, 'output_shape'):
            try:
                layer.output_shape = layer.output.shape
            except AttributeError:
                pass

    keras.utils.plot_model(
        model,
        to_file=str(output_path),
        show_shapes=True,              # Shows Input/Output tensor dimensions
        show_dtype=False,              # Keeps boxes clean by hiding float32 labels
        show_layer_names=True,         # Shows Conv3D, MaxPooling3D, etc.
        rankdir='TB',                  # Forces Top-to-Bottom vertical layout
        expand_nested=True,
        dpi=200,                       # High resolution
        show_layer_activations=True    # Explicitly lists relu, sigmoid, etc.
    )
