import json
import os
import re # regular expressions
import nibabel as nib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from termcolor import colored
from tqdm import tqdm
from scipy import ndimage
from scipy.ndimage import label, center_of_mass
import tensorflow as tf
from keras.utils import Sequence
import keras # NOTE use this instead of tf.keras, for the annoying warnings, try later
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils.class_weight import compute_class_weight
import random

# for tumor diameter calculation
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist

def get_instance_numpy(filepath, verbose=False):
    """
    Returns the .nii file as a numpy nd array.
    """
    # https://nipy.org/nibabel/nibabel_images.html

    if verbose:
        print(f"getting file from {filepath}.")

    # We load the file to create a nibabel image object:
    img = nib.load(filepath)
    
    # Getting the image data "the easy way"
    image_data = img.get_fdata()
    # <class 'numpy.ndarray'>
    # the data is of schape (slice, x, y)
    if verbose:
        print(f"The data is converted into an numpy array of shape {image_data.shape}")
    return image_data

def calculate_max_diameter(mask, zooms, mask_value=2):
    """
    Caslculates the maximum 3D diameter of a segmentation mask (val = mask_value) in millimeters.
    """
    # Find all coordinates where val = 2
    coords = np.argwhere(mask==mask_value)

    if len(coords) == 0: 
        return 0.0
    
    # convert the coordinates to physical millimeters
    coords_mm = coords * np.array(zooms)

    # check for flat dimensions (e.g. tumor exists on a single z-slice)
    valid_dims = [i for i in range(3) if len(np.unique(coords[:, i])) > 1]

    if len(valid_dims) == 0:
        return 0.0 # tumor is a single pixel

    try:
        if len(valid_dims) < 3:
            # 2D Convex hull prevents QHullError if points are prefectly flat
            hull = ConvexHull(coords_mm[:, valid_dims])
        else:
            # 3D Convex hull
            hull = ConvexHull(coords_mm)

        # Get only the outermost vertices to save CPU time
        hull_points = coords_mm[hull.vertices]

        # Calculate max distance between hull vertices
        max_diameter = np.max(pdist(hull_points))
    except Exception as e:
        # Fallback: if Qhull fails for any reason
        # Subsample to 2000 points to prevent OOM errors with pdist

        if len(coords_mm) > 2000:
            idx = np.random.choice(len(coords_mm), 2000, replace=False)
            max_diameter = np.max(pdist(coords_mm[idx]))
        else:
            max_diameter = np.max(pdist(coords_mm))

    return round(max_diameter, 2)

# DEPRECATED - used previously with the much smaller dataset
def preprocess_data(kits_dir, output_dir, annotation_path, valid_cases_path, verbose=False):
    """
    Preprocesses kits23 data, with the additional annotation provided by Medical professionals
    and a list of valid cases (contatining one tumor and no cysts).
    """
    kits_path = Path(kits_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # load data
    with open(valid_cases_path, 'r') as f:
        valid_cases = set(json.load(f))
    
    print("Validating cases against xlsx...")

    labels = []
    df = pd.read_excel(annotation_path)

    required_columns = ['case_id', 'position of tumor', 'Invasion of the renal sinus', 'radius']
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise KeyError(colored(f"CRITICAL ERROR: CSV is missing columns: {missing_cols}", "red", attrs=['bold']))

    # check for cases that are annotated but not in the valid cases
    annotated_but_invalid = df[
        (~df['case_id'].isin(valid_cases)) & 
        (# check against all valid columns
        (df['position of tumor'].notna()) & (df['position of tumor'] != "") |
        (df['Invasion of the renal sinus'].notna()) & (df['Invasion of the renal sinus'] != "") |
        ((df['radius'].notna()) & (df['radius'] != ""))
        )
    ]['case_id'].tolist()

    if annotated_but_invalid:
        print(colored(f"WARNING: {len(annotated_but_invalid)} cases are annotated but missing from valid_cases: {annotated_but_invalid}", "magenta"))

    # setup the progress bar:
    pbar = tqdm(df.iterrows(), total=len(df), desc="Preprocessing KiTS Dataset")

    # validation checks -- case exists both in annotation and in valid cases list
    for _, row in pbar:
        # Iterate over rows in the annotation csv
        case_id = row['case_id']

        if case_id not in valid_cases:
            continue

        position = str(row['position of tumor']).strip() if pd.notna(row['position of tumor']) else ""
        invasion = str(row['Invasion of the renal sinus']).strip() if pd.notna(row['Invasion of the renal sinus']) else ""
        
        if not position or not invasion:
            pbar.write(colored(f"Skipped {case_id}: Missing annotation.", "yellow"))
            continue

        seg_path = kits_path / case_id / "segmentation.nii.gz"
        if not seg_path.exists():
            pbar.write(colored(f"Skipped {case_id}: Segmentation file not found at {seg_path}", "red"))
            continue

        if verbose: pbar.write(f"Processing {case_id}...")

        # process nifti file
        img = nib.load(str(seg_path))
        segmentation_array = img.get_fdata().astype(np.int8) # int8 to save disk space
        zooms = img.header.get_zooms()

        # calculate the tumor diameter
        diameter_mm = calculate_max_diameter(segmentation_array, zooms)

        # Handle the radius points
        radius_annotation = int(row['radius']) if not pd.isna(row['radius']) else np.nan

        # Save the mask as a numpy array
        npy_filename = f"{case_id}.npy"
        np.save(out_path / npy_filename, segmentation_array)
        if verbose: print(f"\t-> Saved mask to {npy_filename}")

        # append labels
        labels.append({
            'case_id': case_id,
            'position_of_tumor': str(position).strip(),
            'invasion_of_renal_sinus': str(invasion).strip(),
            'radius_annotation': radius_annotation,
            'calculated_diameter_mm': diameter_mm,
            'spacing_z': float(zooms[0]),
            'spacing_y': float(zooms[1]),
            'spacing_x': float(zooms[2])
        })
        if verbose: print(f"\t-> labels appended")

    # save labels
    labels_df = pd.DataFrame(labels)
    labels_df.to_csv(out_path/"labels.csv", index=False)

    print(colored("\n--- Data Consistency Summary ---", "cyan", attrs=['bold']))
    for col in ['position_of_tumor', 'invasion_of_renal_sinus', 'radius_annotation']:
        unique_vals = labels_df[col].unique()
        print(f"Unique categories in '{col}': {unique_vals}")

    # Check for radius distribution
    print(f"Number of cases with NaN original radius: {labels_df['radius_annotation'].isna().sum()}")

    print(colored(f"\nProcessing Complete!", "green", attrs=['bold']), f"labels and segmentations for {len(labels_df)} cases saved to {out_path}")
    return

def get_case_path(dataset_filepath: str, instance: int):
    return str(dataset_filepath) + "/case_" +str(instance).zfill(5)

def compute_radius_from_diameter(df):
    """
    Computes the radius points, based on the calculated diameter within the labels.
    """
    bins = [0, 25, 50, np.inf]
    labels = [0, 1, 2]

    # calculate the points
    df['computed_radius_points'] = pd.cut(
        df['calculated_diameter_mm'],
        bins = bins,
        labels = labels,
        include_lowest = True, # Ensures 0 is included
        right = True # Ensures ranges are right-closed 
    )

    # convert to int
    df['computed_radius_points'] = df['computed_radius_points'].astype(int)

def get_best_annotation_files(files, prefix):
    """
    Finds annotations for all cases with a given prefix.
    Can be used to find path to segmentations, like kidneys and/or tumors.        
    """
    pattern = re.compile(rf"{prefix}_instance-(\d+)_annotation-(\d+)\.nii\.gz")
    grouped_instances = {}

    for f in files:
        match = pattern.match(f)
        if match:
            inst_id = int(match.group(1))
            ann_id = int(match.group(2))

            if inst_id not in grouped_instances:
                grouped_instances[inst_id] = []

            grouped_instances[inst_id].append((ann_id, f))

    # for each insance, sort by annotation ID and return the lowest one
    # for consistency
    best_files = []
    for inst_id, anns in grouped_instances.items():
        anns.sort(key=lambda x: x[0])
        best_files.append(anns[0][1])

    return best_files

def extract_tumor_instances(kits_dir):
    """
    Iterates through all kits23 cases, isolates every tumor instance (lowwest existing annotation number),
    calculates their physical diameter, and returns a DataFrame.
    This function is ran to create the dataFrame with all the tumor instances, to create a Google sheet for annotation.
    """
    kits_path = Path(kits_dir)
    data = []

    case_dirs = sorted([d for d in kits_path.iterdir() if d.is_dir() and d.name.startswith("case_")])

    pbar = tqdm(case_dirs, desc="Extracting tumor instance data")

    for case_dir in pbar:
        case_id = case_dir.name
        instance_path = case_dir / 'instances'

        # skip if instance path doesnt exist
        if not instance_path.exists():
            pbar.write(colored(f"WARNING: instance directory doesn't exist in {case_id}", 'red'))
            continue

        all_files = os.listdir(instance_path)
        kidney_files = get_best_annotation_files(all_files, "kidney")
        tumor_files = get_best_annotation_files(all_files, "tumor")

        # if there are no tumors , move on to the next case
        if not tumor_files:
            pbar.write(colored(f'WARNING: no tumor instances in {instance_path}', 'yellow'))
            continue

        # Process kidney instances cache filename, center of mass
        kidneys_data = []
        for k_file in kidney_files:
            try:
                k_path = instance_path / k_file
                k_img = nib.load(str(k_path))
                k_mask = k_img.get_fdata().astype(np.int8)
                k_zooms = k_img.header.get_zooms()

                # Calculate center of mass for the kidney
                k_com_voxel = center_of_mass(k_mask == 1)

                # If mask is completely empty, center_of_mass returns nan or a tuple containing nans
                if k_com_voxel is None or np.isnan(k_com_voxel).any():
                    pbar.write(colored(f"WARNING: Empty mask found for {case_id}/{k_file}", 'yellow'))
                    continue

                # convert voxel coordinates to physical coordinates
                k_com_physical = np.array(k_com_voxel) * np.array(k_zooms)

                kidneys_data.append({
                    'filename': k_file,
                    'com_physical': k_com_physical
                })

            except Exception as e:
                pbar.write(colored(f"ERROR: processing kidney {case_id}/{k_file}: {e}", "red"))

        # process tumors
        for t_file in tumor_files:
            try:
                t_path = instance_path / t_file
                t_img = nib.load(str(t_path))
                t_mask = t_img.get_fdata().astype(np.int8)
                t_zooms = t_img.header.get_zooms()

                if t_mask.max() != 1:
                    pbar.write(colored(f"Skipped {case_id}/{t_file}: max val in the mask not equal 1", "yellow"))
                    continue

                # Calculate center of mass for the tumor
                t_com_voxel = center_of_mass(t_mask == 1)

                # If mask is completely empty, center_of_mass returns nan or a tuple containing nans
                if t_com_voxel is None or np.isnan(t_com_voxel).any():
                    pbar.write(colored(f"WARNING: Empty mask found for {case_id}/{t_file}", 'yellow'))
                    continue

                t_com_physical = np.array(t_com_voxel) * np.array(t_zooms)

                # Find the closest kidney
                closest_kidney = None
                min_distance = float('inf')

                for k_data in kidneys_data:
                    # calculate Euclidian distance between centers of mass
                    distance = np.linalg.norm(t_com_physical - k_data['com_physical'])
                    if distance < min_distance:
                        min_distance = distance
                        closest_kidney = k_data['filename']

                # calculate diameter
                diameter_mm = calculate_max_diameter(t_mask, t_zooms, mask_value=1)

                # get the instance number
                instance_num = re.search(r"_instance-(\d+)", t_file).group(1)

                # save relevant data
                data.append({
                    'tumor_id': f"{case_id}_t{instance_num}",
                    'case_id': case_id,
                    'tumor_instance': t_file,
                    'kidney_instance': closest_kidney,
                    'calculated_diameter_mm': diameter_mm,
                    'spacing_z': float(t_zooms[0]),
                    'spacing_y': float(t_zooms[1]),
                    'spacing_x': float(t_zooms[2]),
                })

            except Exception as e:
                pbar.write(colored(f"ERROR: unknown error processing {case_id}/{t_file}: {e}", "red"))

    df = pd.DataFrame(data)
    return df

def extract_tumor_kidney_bounding_boxes(csv_path, kits_dir):
    """
    Reads the tumor instances from the csv.
    For each one (in the csv file), calculates the bounding boxes of the kidney-tumor system.
    """
    df = pd.read_csv(csv_path)
    kist_path = Path(kits_dir)

    # make sure tumor id is used correctly
    df = df.set_index('tumor_id')

    bbox_data = []

    pbar = tqdm(df.iterrows(), total=len(df), desc="Calculating bounding boxes")

    for tumor_id, row in pbar:
        case_id = row['case_id']
        tumor_file = row['tumor_instance']
        kidney_file = row['kidney_instance']

        # check if all the data exist in the dataFrame
        if pd.isna(kidney_file) or pd.isna(tumor_file) or pd.isna(case_id):
            pbar.write(colored(f"ERROR: missing data in {tumor_id}. Skipping.", "yellow"))
            continue

        # calculate paths
        instance_path = kist_path / case_id / 'instances'
        k_path = instance_path / kidney_file
        t_path = instance_path / tumor_file

        # make sure the files exist
        if not k_path.exists() or not t_path.exists():
            pbar.write(colored(f"ERROR: Missing files for {tumor_id}:\n{k_path} or\n{t_path}. Skipping.", "yellow"))
            continue

        try:
            k_mask = get_instance_numpy(k_path).astype(np.int8)
            t_mask = get_instance_numpy(t_path).astype(np.int8)

            k_coords = np.where(k_mask == 1)
            t_coords = np.where(t_mask == 1)

            # combine the coordinates
            z_coords = np.concatenate([k_coords[0], t_coords[0]])
            y_coords = np.concatenate([k_coords[1], t_coords[1]])
            x_coords = np.concatenate([k_coords[2], t_coords[2]])

            # make sure masks are not empyt

            if len(z_coords) == 0:
                pbar.write(colored(f"ERROR: Both masks are empty for {tumor_id}. Skipping", "yellow")) 

            # get the min and max values for the bounding box

            bbox_data.append({
                'tumor_id': tumor_id,
                'size_z': z_coords.max() - z_coords.min(),
                'size_y': y_coords.max() - y_coords.min(),
                'size_x': x_coords.max() - x_coords.min()
            })

        except Exception as e:
            pbar.write(colored(f"ERROR: Failed processing {tumor_id}: {e}", "red"))

    bbox_df = pd.DataFrame(bbox_data)
    # bbox_df.set_index('tumor_id', inplace=True)

    return bbox_df

def extract_kidney_bounding_boxes(csv_path, kits_dir):
    """
    Reads the tumor instances from the csv.
    For each one (in the csv file), calculates the bounding boxes of the kidney only.
    """
    df = pd.read_csv(csv_path)
    kits_path = Path(kits_dir)

    # make sure tumor id is used correctly
    df = df.set_index('tumor_id')

    bbox_data = []

    pbar = tqdm(df.iterrows(), total=len(df), desc="Calculating kidney bounding boxes")

    for tumor_id, row in pbar:
        case_id = row['case_id']
        tumor_file = row['tumor_instance'] 
        kidney_file = row['kidney_instance']

        # check if all the data exist in the dataFrame
        if pd.isna(kidney_file) or pd.isna(tumor_file) or pd.isna(case_id):
            pbar.write(colored(f"ERROR: missing data in {tumor_id}. Skipping.", "yellow"))
            continue

        # calculate paths
        instance_path = kits_path / case_id / 'instances'
        k_path = instance_path / kidney_file

        # make sure the file exists (removed tumor path check since it's not loaded)
        if not k_path.exists():
            pbar.write(colored(f"ERROR: Missing kidney file for {tumor_id}:\n{k_path}. Skipping.", "yellow"))
            continue

        try:
            # Load only the kidney mask
            k_mask = get_instance_numpy(k_path).astype(np.int8)

            k_coords = np.where(k_mask == 1)

            z_coords = k_coords[0]
            y_coords = k_coords[1]
            x_coords = k_coords[2]

            # make sure mask is not empty
            if len(z_coords) == 0:
                pbar.write(colored(f"ERROR: Kidney mask is empty for {tumor_id}. Skipping.", "yellow")) 
                continue # Added missing continue to prevent max/min crash below

            # get the sizes for the bounding box
            bbox_data.append({
                'tumor_id': tumor_id,
                'size_z': z_coords.max() - z_coords.min(),
                'size_y': y_coords.max() - y_coords.min(),
                'size_x': x_coords.max() - x_coords.min()
            })

        except Exception as e:
            pbar.write(colored(f"ERROR: Failed processing {tumor_id}: {e}", "red"))

    bbox_df = pd.DataFrame(bbox_data)

    return bbox_df

### IMAGE PROCESSING

def process_kidney_tumor_masks(kidney_mask, tumor_mask, target_shape, preserve_geometry):
    """
    Combines, crops, resizes, and pads kidney-tumor masks.
    Centers the mask around a kidney-tumor bbox.
    Returns a one-hot encoded array of shape (Z, Y, X, 3).
    The third dimesnion represents (per layer): background, kidney, tumor.
    """

    # combine masks
    combined = np.zeros_like(kidney_mask, dtype=np.int8)
    combined[kidney_mask == 1] = 1
    combined[tumor_mask == 1] = 2

    # extract the bounding box of the combined system
    z_coords, y_coords, x_coords = np.where(combined > 0)
    if len(z_coords) == 0:
        return np.zeros(target_shape + (3,), dtype=np.int8) # Empty output as fallback
    
    cropped = combined[
        z_coords.min():z_coords.max()+1,
        y_coords.min():y_coords.max()+1,
        x_coords.min():x_coords.max()+1
    ]

    # Calculate zoom factors
    current_shape = cropped.shape
    zoom_factors = [t / c for t, c in zip(target_shape, current_shape)]

    if preserve_geometry:
        # Scale all axes by the most constraining facor, such that the geometry is preserved
        min_zoom = min(zoom_factors)
        zoom_factors = (min_zoom, min_zoom, min_zoom)

    # Resize the array
    resized = ndimage.zoom(cropped, zoom_factors, order=0, mode='constant', cval=0)

    # slice away any accidental extra voxels (floating point overshoot)
    resized = resized[:target_shape[0], :target_shape[1], :target_shape[2]]

    # Pad to target shape (this should only go into play if geometry is preserved)
    pad_z = max(0, target_shape[0] - resized.shape[0])
    pad_y = max(0, target_shape[1] - resized.shape[1])
    pad_x = max(0, target_shape[2] - resized.shape[2])

    # use symmetric padding
    pads = (
        (pad_z // 2, pad_z - pad_z //2),
        (pad_y // 2, pad_y - pad_y //2),
        (pad_x // 2, pad_x - pad_x //2)
    )

    padded = np.pad(resized, pads, mode='constant', constant_values=0)

    if padded.shape != target_shape:
        raise ValueError(f"Unexpected output shape. ", f"Expected {target_shape}, got {padded.shape}")

    # apply one-hot encoding
    one_hot_mask = keras.utils.to_categorical(padded, num_classes=3)
        
    return one_hot_mask

def process_kidney_centered_masks(kidney_mask, tumor_mask, target_shape, preserve_geometry, margin=0.1):
    """
    Combines, crops, resizes, and pads kidney-tumor masks.
    Centers the crop window strictly around the kidney bbox, expanded by a margin.
    """
    # Combine the masks
    combined = np.zeros_like(kidney_mask, dtype=np.int8)
    combined[kidney_mask == 1] = 1
    combined[tumor_mask == 1] = 2

    # Extract kidney bounding box
    z_coords, y_coords, x_coords = np.where(kidney_mask == 1)
    if len(z_coords) == 0:
        return np.zeros(target_shape + (3,), dtype=np.int8) # Fallback if no kidney found
    
    # Calculate raw kidney dimensions
    len_z = z_coords.max() - z_coords.min() + 1
    len_y = y_coords.max() - y_coords.min() + 1
    len_x = x_coords.max() - x_coords.min() + 1

    # Compute margins
    margin_z = int(np.floor(len_z * margin))
    margin_y = int(np.floor(len_y * margin))
    margin_x = int(np.floor(len_x * margin))

    # Expand bounding box and clip to ensure we stay within physical image boundaries
    z_start = max(0, z_coords.min() - margin_z)
    z_end = min(combined.shape[0], z_coords.max() + 1 + margin_z)

    y_start = max(0, y_coords.min() - margin_y)
    y_end = min(combined.shape[1], y_coords.max() + 1 + margin_y)

    x_start = max(0, x_coords.min() - margin_x)
    x_end = min(combined.shape[2], x_coords.max() + 1 + margin_x)

    # Crop the combined matrix
    cropped = combined[z_start:z_end, y_start:y_end, x_start:x_end]

    # Calculate zoom factors
    current_shape=cropped.shape
    zoom_factors = [t / c for t, c in zip(target_shape, current_shape)]

    if preserve_geometry:
        min_zoom = min(zoom_factors)
        zoom_factors = (min_zoom, min_zoom, min_zoom)

    # Resize the array
    resized = ndimage.zoom(cropped, zoom_factors, order=0, mode='constant', cval=0)
    resized = resized[:target_shape[0], :target_shape[1], :target_shape[2]]

    # Symmetric padding (active if preserve_geometry is True)
    pad_z = max(0, target_shape[0] - resized.shape[0])
    pad_y = max(0, target_shape[1] - resized.shape[1])
    pad_x = max(0, target_shape[2] - resized.shape[2])

    pads = (
        (pad_z // 2, pad_z - pad_z // 2),
        (pad_y // 2, pad_y - pad_y // 2),
        (pad_x // 2, pad_x - pad_x // 2)
    )

    padded = np.pad(resized, pads, mode='constant', constant_values=0)

    if padded.shape != target_shape:
        raise ValueError(f"Unexpected output shape. Expected {target_shape}, got {padded.shape}")
    
    # One-hot encoding
    one_hot_mask = keras.utils.to_categorical(padded, num_classes=3)
    return one_hot_mask

def prepare_cnn_data(df, kits_dir, out_dir, target_shape, bbox:str, preserve_geometry, margin=0.1):
    """
    Prepares the dataset for ML, verifies class limits, and documents column orders.
    """
    kits_path = Path(kits_dir)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Export dataset metadata
    dataset_metadata = {
        "target_shape": list(target_shape),  # Convert tuple to list for proper JSON formatting
        "bbox": bbox,
        "preserve_geometry": bool(preserve_geometry),
        "margin": float(margin) if bbox == 'kidney' else None
    }

    with open(out_path / "dataset_metadata.json", "w") as f:
        json.dump(dataset_metadata, f, indent=4)

    # drop NA
    df = df.dropna(subset=['case_id', 'tumor_instance', 'kidney_instance'])

    # explicit class check
    unique_positions = df['position'].dropna().unique()
    unique_invasions = df['invasion'].dropna().unique()

    if len(unique_positions) > 3:
        raise ValueError(f"Data constraint violation! Expected max 3 unique positions, found {len(unique_positions)}: {unique_positions}")
    if len(unique_invasions) > 3:
        raise ValueError(f"Data constraint violation! Expected max 3 unique invasions, found {len(unique_invasions)}: {unique_invasions}")

    # one hot encode the tabular labels
    df = pd.get_dummies(df, columns=['position', 'invasion'])

    # Lock in strict alphabetical order
    pos_cols = sorted([col for col in df.columns if col.startswith('position_')])
    inv_cols = sorted([col for col in df.columns if col.startswith('invasion_')])
    tabular_cols = pos_cols + inv_cols 

    # Export column structure map alongside data arrays
    with open(out_path / "tabular_columns.txt", "w") as f:
        f.write("\n".join(tabular_cols))

    pbar = tqdm(df.iterrows(), total=len(df), desc="Preparing CNN data")

    for idx, row in pbar:
        tumor_id = row['tumor_id']
        case_id = row['case_id']

        if pd.isna(row['tumor_instance']) or pd.isna(row['kidney_instance']):
            continue

        # Load masks
        instance_path = kits_path / case_id / 'instances'
        k_path = instance_path / row['kidney_instance']
        t_path = instance_path / row['tumor_instance']

        if not k_path.exists() or not t_path.exists():
            continue

        # get the masks
        k_mask = nib.load(str(k_path)).get_fdata()
        t_mask = nib.load(str(t_path)).get_fdata()

        # process the masks
        if bbox == 'kidney-tumor':
            final_mask = process_kidney_tumor_masks(k_mask, t_mask, target_shape, preserve_geometry)
        elif bbox == 'kidney':
            final_mask = process_kidney_centered_masks(k_mask, t_mask, target_shape, preserve_geometry, margin)
            # NOTE remember to copy the margin value for translation range in the generator
        else:
            raise ValueError(
                f"Invalid configuration for argument 'bbox'. "
                f"Received '{bbox}', but expected either 'kidney-tumor' or 'kidney'."
                )

        # Isolate target numerical arrays
        tabular_features = row[tabular_cols].astype(np.float32).values

        # save the .npy files (masks)
        case_out_dir = out_path / tumor_id
        case_out_dir.mkdir(exist_ok=True)

        np.save(case_out_dir / "mask.npy", final_mask)
        np.save(case_out_dir / "labels.npy", tabular_features)

def generate_train_validation_split(df, n_splits=5): 
    """
    Splits the dataset into K-folds, ensuring no patient leakage (grouped by case_id)
    and preserving the target distribution (Stratified by position).
    Returns a dataframe containing the split mapping (tumor_id, case_id, fold). 
    """
    # Create a copy and initialize our blank canvas column
    df_splits = df.copy()
    df_splits['fold'] = -1

    # Targets for stratification and groups for leakage prevention
    y = df_splits['position'].values
    groups = df_splits['case_id'].values

    SGKF = StratifiedGroupKFold(n_splits=n_splits)

    for fold, (train_idx, val_idx) in enumerate(SGKF.split(df_splits, y, groups)):
        # Assign the current fold number to the validation indices
        df_splits.iloc[val_idx, df_splits.columns.get_loc('fold')] = fold

        # Print diagnostic fold data
        print(f"\n--- Fold {fold} Created ---")
        print(f"\tTrain samples: {len(train_idx)} | Validation samples: {len(val_idx)}")
        print(f"\tVal Class Dist: {pd.Series(y[val_idx]).value_counts(normalize=True).to_dict()}")

    # Return only the necessary mapping columns
    return df_splits[['tumor_id', 'case_id', 'fold']]

### GENERATOR

from keras.utils import PyDataset
class KITSDataGenerator(PyDataset):
    def __init__(self, paths, batch_size=4,
                 augment=False, augment_enable=['flip','translate','zoom'],
                 max_translation=[0,0,0],
                 zoom_range=[0.9,1.1],
                 shuffle=True, workers=1, use_multiprocessing=False):
        
        self.paths = np.array(paths)
        self.batch_size = int(batch_size)
        self.augment = augment
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.paths))

        # Pass processing settings directly to the Keras data handler class
        super().__init__(workers=workers, use_multiprocessing=use_multiprocessing, max_queue_size=10)

        self.on_epoch_end() 

        # Augmentation hyperparameters
        self.augment_enable = set(augment_enable) 
        self.max_translation = max_translation
        self.zoom_range = zoom_range

    def __len__(self):
        return int(np.ceil(len(self.paths) / self.batch_size))
    
    def __getitem__(self, index):
        batch_indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]
        batch_paths = self.paths[batch_indexes]
        X, y = self.__data_generation(batch_paths)
        return X, y
    
    def on_epoch_end(self):
        self.indexes = np.arange(len(self.paths))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, batch_paths):
        X, y = [], []
        for path_str in batch_paths:
            path = Path(path_str)
            mask = np.load(path / "mask.npy")
            tabular = np.load(path / "tabular.npy")

            if self.augment:
                mask = self._augment_3d_volume(mask)

            X.append(mask)
            y.append(tabular)

        return np.array(X), np.array(y)

    def _augment_3d_volume(self, volume):
        if 'flip' in self.augment_enable:
            if random.random() > 0.5:
                volume = np.flip(volume, axis=1) 
            if random.random() > 0.5:
                volume = np.flip(volume, axis=2) 

        if 'translate' in self.augment_enable and random.random() > 0.5:
            shift_z = random.randint(-self.max_translation[0], self.max_translation[0])
            shift_y = random.randint(-self.max_translation[1], self.max_translation[1])
            shift_x = random.randint(-self.max_translation[2], self.max_translation[2])
            volume = ndimage.shift(volume, shift=(shift_z, shift_y, shift_x, 0), order=0, mode='constant', cval=0.0)
        
        if 'zoom' in self.augment_enable and random.random() > 0.5:
            scale = random.uniform(self.zoom_range[0], self.zoom_range[1])
            zoomed = ndimage.zoom(volume, zoom=(scale, scale, scale, 1), order=0, mode='constant', cval=0.0)
            volume = self._crop_or_pad_to_shape(zoomed, target_shape=volume.shape)

        missing_background = np.all(volume == 0, axis=-1)
        volume[missing_background, 0] = 1.0
        return volume
    
    def _crop_or_pad_to_shape(self, volume, target_shape):
        out_volume = np.zeros(target_shape, dtype=volume.dtype)
        v_shape = volume.shape[:-1]
        t_shape = target_shape[:-1]

        start_v = [max(0, (v - t) // 2) for v, t in zip(v_shape, t_shape)]
        start_t = [max(0, (t - v) // 2) for v, t in zip(v_shape, t_shape)]
        lengths = [min(v, t) for v, t in zip(v_shape, t_shape)]
        
        out_volume[
            start_t[0]:start_t[0]+lengths[0],
            start_t[1]:start_t[1]+lengths[1],
            start_t[2]:start_t[2]+lengths[2],
            :
        ] = volume[
            start_v[0]:start_v[0]+lengths[0],
            start_v[1]:start_v[1]+lengths[1],
            start_v[2]:start_v[2]+lengths[2],
            :
        ] 
        return out_volume

def get_train_validation_paths(dataset_dir, fold_no):
    dataset_path = Path(dataset_dir)
    splits_csv_path = dataset_path / "dataset_splits.csv"

    if not splits_csv_path.exists():
        raise FileNotFoundError(f"Could not find splits mapping at {splits_csv_path}")
    
    df_splits = pd.read_csv(splits_csv_path)
    df_val = df_splits[df_splits['fold'] == fold_no]
    df_train = df_splits[df_splits['fold'] != fold_no]

    train_paths = [str(dataset_path / tumor_id) for tumor_id in df_train['tumor_id']]
    val_paths = [str(dataset_path / tumor_id) for tumor_id in df_val['tumor_id']]
    print(f"Loaded paths for Fold {fold_no} from {str(dataset_path)}")
    return train_paths, val_paths

def get_keras_class_weights(train_labels):
    """
    Compute class weights inversly proportional to class frequencies.
    To combat overfitting to more common cases.
    """
    classes = np.unique(train_labels)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=train_labels)

    class_weight_dict = {i: float(w) for i, w in enumerate(weights)}
