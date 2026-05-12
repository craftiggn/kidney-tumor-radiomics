import json
import nibabel as nib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from termcolor import colored
from tqdm import tqdm

# for tumor diameter calculation
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist

def get_instance_numpy(filepath):
    """
    Returns the .nii file as a numpy nd array.
    """
    # https://nipy.org/nibabel/nibabel_images.html

    print(f"getting file from {filepath}.")

    # We load the file to create a nibabel image object:
    img = nib.load(filepath)
    
    # Getting the image data "the easy way"
    image_data = img.get_fdata()
    # <class 'numpy.ndarray'>
    # the data is of schape (slice, x, y)
    print(f"The data is converted into an numpy array of shape {image_data.shape}")
    return image_data

def calculate_max_diameter(mask, zooms):
    """
    Caslculates the maximum 3D diameter of a segmentation mask (val = 2) in millimeters.
    """
    # Find all coordinates where val = 2
    coords = np.argwhere(mask==2)

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



# use keras layers api 
# https://keras.io/api/layers/preprocessing_layers/

# NOTE
# add a function to analyze the labels and separate it into train/ split
# when loading the dataset use np.load(f"{output_dir}/{case_id}.npy", mmap_mode='r')
# for the fastest possible data reading

