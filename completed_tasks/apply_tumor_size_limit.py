import pandas as pd

TUMOR_INSTANCE_PATH = 'data/kits23_tumor_instance_data.csv'
TUMOR_SIZE_LIMIT_MM  = 75
ANNOTATION_PATH = f'data/annotation_empty_{TUMOR_SIZE_LIMIT_MM}mm.csv'

df = pd.read_csv(TUMOR_INSTANCE_PATH)
df = df[df['calculated_diameter_mm'] <= TUMOR_SIZE_LIMIT_MM]

# some tumors do not fit the RPN requirements
cases_to_drop = [
    'case_00458',
    'case_00466',
    'case_00517',
    'case_00524'
]

df = df[~df['case_id'].isin(cases_to_drop)]

df.to_csv(ANNOTATION_PATH, index=False)