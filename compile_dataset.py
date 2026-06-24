import pandas as pd
from pathlib import Path
from modules import data_preprocessing as dataPrep
from modules import plotting_functions as plf

# --- CONFIGURATIONS ---
ANNOTATION_PATH = 'data/annotations_RPN_75mm_final.csv'
KITS23_DATASET_PATH = '/home/marcel/Repos/Data/kits23'
BASE_OUTPUT_DIR = Path('/home/marcel/Repos/Data')

# Define target experiments to generate
TARGET_SHAPES = [(123, 123, 116), ] 
BBOX_STRATEGIES = ['kidney']
N_SPLITS = 5

# --- LOAD & CLEAN DATA ---
df = pd.read_csv(ANNOTATION_PATH)
df = df[df['approved_for_training']]  # remove rejected cases
df = df.dropna(subset=['case_id', 'tumor_instance', 'kidney_instance', 'position', 'invasion', 'radius_pts'])


# --- GENERATE DETERMINISTIC K-FOLD SPLITS ---
# Calls our refactored function to generate the master split map
print("Generating Stratified Group K-Fold Splits...")
df_master_splits = dataPrep.generate_train_validation_split(df, n_splits=N_SPLITS)

# Save this out as a master split reference for audit trails
df_master_splits.to_csv(BASE_OUTPUT_DIR / "kfold_master_splits.csv", index=False)


# --- LOOP AND COMPILE DATASETS ---
for target_shape in TARGET_SHAPES:
    for bbox_strat in BBOX_STRATEGIES:
        
        # Generate shape description string
        shape_str = "_".join(str(n) for n in target_shape)
        dir_name = f"kits_dataset_{shape_str}_{bbox_strat}"
        
        compiled_dataset_dir = BASE_OUTPUT_DIR / dir_name
        dataset_figures_dir = BASE_OUTPUT_DIR / f"{dir_name}_figures"
        
        print(f"\nProcessing: {dir_name}")
        
        # Process arrays and store files
        dataPrep.prepare_cnn_data(
            df=df.copy(), 
            kits_dir=KITS23_DATASET_PATH, 
            out_dir=compiled_dataset_dir, 
            target_shape=target_shape, 
            bbox=bbox_strat, 
            preserve_geometry=False
        )
        
        # Save a localized copy of the split map matching exactly what's inside this directory
        df_master_splits.to_csv(compiled_dataset_dir / "dataset_splits.csv", index=False)
        
        # --- 4. GENERATE DIAGNOSTIC VIDEOS ---
        print(f"Generating tumor verification videos...")
        dirs = [p for p in compiled_dataset_dir.iterdir() if p.is_dir()]
        for d in dirs:
            plf.generate_tumor_video_from_npy(d, dataset_figures_dir)

print("\nDataset compilation pipeline complete!")