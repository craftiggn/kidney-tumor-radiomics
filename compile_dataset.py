from modules import data_preprocessing as dataPrep
from modules import plotting_functions as plf
import pandas as pd
from pathlib import Path

ANNOTATION_PATH = 'data/annotations_RPN_75mm_final.csv'
KITS23_DATASET_PATH = '/home/marcel/Repos/Data/kits23'

TARGET_SHAPE =  (123, 123, 116)

dir_name_default = "kits_dataset_"
dir_name_default += "_".join(str(n) for n in TARGET_SHAPE)


COMPILED_DATASET_DIR_TRAINING = '/home/marcel/Repos/Data/' + dir_name_default + "_training"
COMPILED_DATASET_DIR_VALIDATION = '' # leave empty for now

DATASET_FIGURES_DIR = '/home/marcel/Repos/Data/' + dir_name_default + "_figures"

# SPLIT THE DATA INTO A TRAINING AND VALIDATION DATASET

# GENERATE THE DATASETS
df = pd.read_csv(ANNOTATION_PATH)
df = df[df['approved_for_training']]
df = df.iloc[:50]
print(df)

dataPrep.prepare_cnn_data(df, KITS23_DATASET_PATH, COMPILED_DATASET_DIR_TRAINING, TARGET_SHAPE)

# GENERATE FIGURES SHOWING THE CHARACTERISTICS OF THE TRAIN AND TEST DATASET

folder = Path(COMPILED_DATASET_DIR_TRAINING)

dirs = [p for p in folder.iterdir() if p.is_dir()]
print(dirs)

for d in dirs:
    plf.generate_tumor_video_from_npy(d, DATASET_FIGURES_DIR)