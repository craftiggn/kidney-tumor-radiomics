import pandas as pd
from pathlib import Path
from termcolor import colored
from modules import data_preprocessing as dataPrep
from modules import plotting_functions as plf

compiled_dataset_dir = Path('/home/marcel/Repos/Data/kits23_processed/kits_dataset_36_60_60_kidney')
print(f"Generating videos...")
dirs = [p for p in compiled_dataset_dir.iterdir() if p.is_dir()]
for d in dirs:
    plf.generate_tumor_video_from_npy(d, '/home/marcel/Desktop/figures_36_60_60_kidney')