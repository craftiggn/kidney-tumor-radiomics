from modules import data_preprocessing as dataPrep

KITS_DIR = '/home/marcel/Repos/Data/kits23'
OUTPUT_DIR = '/home/marcel/Repos/Data/kits23_converted'
ANNOTATION_PATH = 'data/kits23 JSON table - JSON.csv'
VALID_CASES_PATH = 'data/single_tumor_cases.json'

dataPrep.preprocess_data(KITS_DIR, OUTPUT_DIR, ANNOTATION_PATH, VALID_CASES_PATH)