from modules import data_preprocessing as data_prep

KITS_DIR = "/home/marcel/Repos/Data/kits23"
OUTPUT_CSV = "data/kits23_tumor_instance_data.csv"

tumor_df = data_prep.extract_tumor_instances(KITS_DIR)
print(tumor_df.head())
tumor_df.to_csv(OUTPUT_CSV)