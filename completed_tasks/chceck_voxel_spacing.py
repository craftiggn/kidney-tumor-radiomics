import os
import json
import nibabel as nib
from pathlib import Path
from termcolor import colored

def precheck_voxel_spacings(base_dir, json_path, num_samples=10):
    """
    Checks the voxel spacing (zooms) for a few cases to confirm resolution variance.
    """
    base_path = Path(base_dir)
    
    # 1. Load the valid cases
    try:
        with open(json_path, 'r') as f:
            valid_cases = json.load(f)
    except FileNotFoundError:
        print(colored(f"Error: Could not find {json_path}", "red"))
        return

    print(colored(f"Checking voxel spacing for the first {num_samples} valid cases...\n", "cyan", attrs=["bold"]))
    
    cases_checked = 0
    for case_id in valid_cases:
        if cases_checked >= num_samples:
            break
            
        seg_path = base_path / case_id / "segmentation.nii.gz"
        
        if seg_path.exists():
            try:
                # Load the NIfTI object (this reads the header, but not the heavy array yet)
                img = nib.load(seg_path)
                
                # Get shape and physical spacing (zooms)
                shape = img.shape
                zooms = img.header.get_zooms()
                
                print(colored(f"Case: {case_id}", "green", attrs=["bold"]))
                print(f"  Array Shape (voxels):    {shape}")
                
                # NiBabel usually aligns zooms with the shape dimensions
                # Format the zooms to 3 decimal places for readability
                formatted_zooms = tuple(round(z, 3) for z in zooms)
                print(f"  Voxel Spacing (mm):      {formatted_zooms}")
                print("-" * 40)
                
                cases_checked += 1
                
            except Exception as e:
                print(colored(f"Error reading {case_id}: {e}", "red"))
        else:
            print(colored(f"Warning: {seg_path} not found.", "yellow"))

# --- How to run it ---
# Adjust these paths to point to your actual local files
kits_data_path = "/home/marcel/Repos/Data/kits23/"
json_file_path = "data/single_tumor_cases.json"

precheck_voxel_spacings(kits_data_path, json_file_path, num_samples=10)