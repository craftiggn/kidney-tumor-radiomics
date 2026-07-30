import pandas as pd
from pathlib import Path
from modules import data_preprocessing as dataPrep
from modules import plotting_functions as plf
from termcolor import colored

# --- CONFIGURATIONS ---
ANNOTATION_PATH = 'data/annotations_RPN_75mm_final.csv'
# KITS23_DATASET_PATH = '/home/marcel/Repos/Data/kits23'
KITS23_DATASET_PATH = '/home/marcel42/Data/kits23'
# BASE_OUTPUT_DIR = Path('/home/marcel/Repos/Data/kits23_processed')
BASE_OUTPUT_DIR = Path('/home/marcel42/Data/kits23_processed')

# Define target experiments to generate
TARGET_SHAPES = [(64, 123, 116)]
# https://www.ncbi.nlm.nih.gov/books/NBK482385/?utm_source=chatgpt.com
# "Typical dimensions include a length of 10 to 12 cm, a width of 5 to 7 cm, and a thickness of 3 to 5 cm, comparable in size to a closed fist."
# So proportion should be ~11:6:4

# TARGET_SHAPES.append((110, 60, 40))
# TARGET_SHAPES.append((85, 45, 30))
# TARGET_SHAPES.append((55, 30, 20))

# given the often very low resolution in the Z-axis lets try generating datasets with a low Z-axis resolution
TARGET_SHAPES.append((32, 103, 116))
TARGET_SHAPES.append((36, 60, 60))
TARGET_SHAPES.append((16, 30, 30))
BBOX_STRATEGIES = ['kidney']
N_SPLITS = 5

# --- LOAD & CLEAN DATA ---
df = pd.read_csv(ANNOTATION_PATH)
df = df[df['approved_for_training']]  # remove rejected cases
# df = df.iloc[:10] # for prototyping
df = df.dropna(subset=['case_id', 'tumor_instance', 'kidney_instance', 'position', 'invasion', 'radius_pts'])

# --- GENERATE DETERMINISTIC K-FOLD SPLITS ---
df_master_splits = dataPrep.generate_train_validation_split(df, n_splits=N_SPLITS)

# Save this out as a master split reference for audit trails
df_master_splits.to_csv(BASE_OUTPUT_DIR / "kfold_master_splits.csv", index=False)
print(f"\nSaved Stratified Group K-Fold Splits to {BASE_OUTPUT_DIR}/kfold_master_splits.csv")

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
        
        # Save a localized copy of the split map matching exactly what's inside this directory
        df_master_splits.to_csv(compiled_dataset_dir / "dataset_splits.csv", index=False)
        
        # --- GENERATE DIAGNOSTIC VIDEOS ---
        # print(f"Generating videos...")
        # dirs = [p for p in compiled_dataset_dir.iterdir() if p.is_dir()]
        # for d in dirs:
        #     plf.generate_tumor_video_from_npy(d, dataset_figures_dir)

print(colored("\nDataset generation complete.", "blue", attrs=["bold"]))