from modules import data_preprocessing as dataPrep
from modules import plotting_functions as plf
import pandas as pd
from pathlib import Path

ANNOTATION_PATH = 'data/annotations_RPN_75mm_final.csv'
KITS23_DATASET_PATH = '/home/marcel/Repos/Data/kits23'

TARGET_SHAPE =  (123, 123, 116)

dir_name_default = "kits_dataset_"
dir_name_default += "_".join(str(n) for n in TARGET_SHAPE)


COMPILED_DATASET_DIR = '/home/marcel/Repos/Data/' + dir_name_default # add cenetering option here

DATASET_FIGURES_DIR = '/home/marcel/Repos/Data/' + dir_name_default + "_figures"


# GENERATE THE DATASETS
df = pd.read_csv(ANNOTATION_PATH)
df = df[df['approved_for_training']] # remove rejected cases
df = df.dropna(subset=['case_id', 'tumor_instance', 'kidney_instance', 'position', 'invasion', 'radius_pts']) # remove empty cases

df = df.iloc[:50] # temporary 
print(df) # temporary
dataPrep.prepare_cnn_data(df, KITS23_DATASET_PATH, COMPILED_DATASET_DIR, TARGET_SHAPE)
# SPLIT THE DATA INTO A TRAINING AND VALIDATION DATASETS USING K-FOLD SPLIT

# GENERATE FIGURES SHOWING THE CHARACTERISTICS OF THE TRAIN AND TEST DATASET

folder = Path(COMPILED_DATASET_DIR)

dirs = [p for p in folder.iterdir() if p.is_dir()]
print(dirs)

for d in dirs:
    plf.generate_tumor_video_from_npy(d, DATASET_FIGURES_DIR)