from modules import data_preprocessing as dataPrep

KITS_DIR = '/home/marcel/Repos/Data/kits23'
OUTPUT_DIR = '/home/marcel/Repos/Data/kits23_processed'
ANNOTATION_PATH = 'data/kits23_annotation_raw.xlsx'
VALID_CASES_PATH = 'data/single_tumor_cases.json'

dataPrep.preprocess_data(KITS_DIR, OUTPUT_DIR, ANNOTATION_PATH, VALID_CASES_PATH)