import pandas as pd
from pathlib import Path
from termcolor import colored
from modules import data_preprocessing as dataPrep
from modules import plotting_functions as plf

# --- CONFIGURATIONS ---
ANNOTATION_PATH = 'data/annotations_RPN_75mm_final.csv'
# HPC
# KITS23_DATASET_PATH = '/home/marcel42/Data/kits23'
# BASE_OUTPUT_DIR = Path('/home/marcel42/Data/kits23_processed')
# LOCAL
KITS23_DATASET_PATH = '/home/marcel/Repos/Data/kits23'
BASE_OUTPUT_DIR = Path('/home/marcel/Repos/Data/kits23_processed')

# Define target experiments to generate
# TARGET_SHAPES = [(32, 103, 116), (36, 60, 60), (16, 30, 30)]
TARGET_SHAPES = [(36, 60, 60)]

BBOX_STRATEGIES = ['kidney'] # or 'kidney-tumor'
N_SPLITS = 5

# --- LOAD AND CLEAN DATA ---
df = pd.read_csv(ANNOTATION_PATH)
df = df[df['approved_for_training']]  # remove rejected cases
df = df.dropna(subset=['case_id', 'tumor_instance', 'kidney_instance', 'position', 'invasion', 'radius_pts'])

# --- GENERATE K-FOLD SPLITS ---
df_master_splits = dataPrep.generate_train_validation_split(df, n_splits=N_SPLITS)

# --- LOOP AND COMPILE DATASETS ---
print(colored("\nDataset generation:", "blue", attrs=["bold"]))
print(f"Shapes: {TARGET_SHAPES}")
print(f"Bbox strategies: {BBOX_STRATEGIES}\n")

for target_shape in TARGET_SHAPES:
    for bbox_strat in BBOX_STRATEGIES:
        # Generate shape description string
        shape_str = "_".join(str(n) for n in target_shape)
        dir_name = f"kits_dataset_{shape_str}_{bbox_strat}"
        
        compiled_dataset_dir = BASE_OUTPUT_DIR / dir_name
        dataset_figures_dir = BASE_OUTPUT_DIR / f"figures_{dir_name}"
        
        print(f"\nProcessing: {dir_name}")
        
        # Generate the dataset
        dataPrep.prepare_cnn_data(
            df=df.copy(), 
            kits_dir=KITS23_DATASET_PATH, 
            out_dir=compiled_dataset_dir, 
            target_shape=target_shape, 
            bbox=bbox_strat, 
            preserve_geometry=False
        )
        
        # Save a copy of the split map for the generator 
        df_master_splits.to_csv(compiled_dataset_dir / "dataset_splits.csv", index=False)
        
        # --- GENERATE DIAGNOSTIC VIDEOS ---
        if True: # to enable: set to True
            print(f"Generating videos...")
            dirs = [p for p in compiled_dataset_dir.iterdir() if p.is_dir()]
            for d in dirs:
                plf.generate_tumor_video_from_npy(d, dataset_figures_dir)

print(colored("\nDataset generation complete.", "blue", attrs=["bold"]))