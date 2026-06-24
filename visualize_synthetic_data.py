from modules import data_preprocessing as dataPrep
from modules import plotting_functions as plf

DATASET_DIR = '/home/marcel/Repos/Data/kits23_processed/kits_dataset_123_123_116_kidney'
fold_no = 0

train_paths, val_paths = dataPrep.get_train_validation_paths(DATASET_DIR, fold_no)


generator = dataPrep.KITSDataGenerator(train_paths, batch_size=4, augment=True, shuffle=True,
                                       max_translation=[10, 10, 10])

batch_X, batch_y = generator[0]

plf.visualize_generator_batch(
    batch_X=batch_X,
    batch_y=batch_y,
    experiment_dir=DATASET_DIR,
    out_dir="./generator_sanity_checks",
    batch_name_prefix=f"fold_{fold_no}_train"
)