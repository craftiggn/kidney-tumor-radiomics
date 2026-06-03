from modules import data_preprocessing as data_prep
import pandas as pd
import numpy as np

KITS_DIR = "/home/marcel/Repos/Data/kits23"
OUTPUT_CSV = "data/kits23_tumor_instance_data.csv"

# generate the list of cases, their id, radius and spacial parameters for the tumor
tumor_df = data_prep.extract_tumor_instances(KITS_DIR) # Note, modify this such that it features also the kidney annotation
print(tumor_df.head())

tumor_df['position'] = pd.NA
tumor_df['invasion'] = pd.NA
tumor_df['radius_pts'] = np.nan

tumor_df.to_csv(OUTPUT_CSV)