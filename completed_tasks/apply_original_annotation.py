import pandas as pd
from termcolor import colored

RAW_ANNOTATION_PATH = 'data/kits23_annotation_raw - JSON.csv'
TUMOR_INSTANCE_PATH = 'data/kits23_tumor_instance_data.csv'

df1 = pd.read_csv(RAW_ANNOTATION_PATH)

columns_to_keep = ['case_id', 'position of tumor', 'Invasion of the renal sinus', 'radius']

# keep only relevant rows
df1 = df1[columns_to_keep]
# drop NaN values
df1 = df1.dropna()

df2 = pd.read_csv(TUMOR_INSTANCE_PATH)

# need to re-assert data types
df2['position'] = df2['position'].astype('string')
df2['invasion'] = df2['invasion'].astype('string')
df2['radius_pts'] = df2['radius_pts'].astype('Int64')  # Int64 allows integers with NA

# for every row in df1:
# check that it exists in df2, by case_id
# confirm that this case_id exists for only one row in df2, if not throw a warning
# transsfer the annotations to df2

# pre-calculate cunts of case_id in df2 to check for dupllicates
case_counts = df2['case_id'].value_counts()

print(colored("--- Starting annotation transfer ---", 'blue', attrs=['bold']))

for _, row in df1.iterrows():
    case_id = row['case_id']

    # check that it exists in df2
    if case_id not in case_counts:
        continue

    # confirm that this case_id exists for only one fow in df2
    if case_counts[case_id] > 1:
        print(colored(f"WARNING: Multiple tumor instances found for {case_id}. Skipping annotation transfer", 'yellow'))
        continue

    # transfer the annotations to df2
    # find the exact index in df2 where the case_id matches
    target_idx = df2.index[df2['case_id'] == case_id][0]
    
    df2.at[target_idx, 'position'] = row['position of tumor']
    df2.at[target_idx, 'invasion'] = row['Invasion of the renal sinus']
    df2.at[target_idx, 'radius_pts'] = row['radius']
    print(f"Transfer succesfull for {case_id}")

df2.to_csv(TUMOR_INSTANCE_PATH, index=False)