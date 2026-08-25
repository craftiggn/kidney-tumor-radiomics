from modules import data_preprocessing as data_prep
import pandas as pd
import numpy as np
from termcolor import colored

KITS_DIR = "/home/marcel/Repos/Data/kits23"
OUTPUT_CSV = "data/kits23_tumor_instance_data.csv"

# generate the list of cases, their id, radius and spacial parameters for the tumor
tumor_df = data_prep.extract_tumor_instances(KITS_DIR)
print(colored("--- Extracted tumor instances ---", 'blue', attrs=['bold']))
print(tumor_df.head())

# add annotation labels with explicit data types
tumor_df['position'] = pd.Series(pd.NA, index=tumor_df.index, dtype='string')
tumor_df['invasion'] = pd.Series(pd.NA, index=tumor_df.index, dtype='string')
tumor_df['radius_pts'] = pd.Series(pd.NA, index=tumor_df.index, dtype='Int64') # Int64 allows integers with NA
tumor_df['approved_for_training'] = True # by default

print(colored("--- Added annotation labels ---", 'blue', attrs=['bold']))
print(tumor_df.head())


tumor_df.to_csv(OUTPUT_CSV, index=False)