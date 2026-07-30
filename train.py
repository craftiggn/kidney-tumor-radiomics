import os
# Suppress standard TensorFlow INFO and WARNING logs (set to '3' to mute errors too, '2' filters info/warnings)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import tensorflow as tf
import keras
from termcolor import colored
import json
import gc

# import custom modules
from modules import plotting_functions as plf
from modules import data_preprocessing as dataPrep
from modules import model_functions as mod

# sample run command:
# python3 train.py --data_dir /home/marcel/Repos/Data/kits23_processed/kits_dataset_123_123_116_kidney --epochs 10 --n_folds 1

def parse_arguments():
    parser = argparse.ArgumentParser(description="Train 3D CNN on KiTS23 Preprocessed Dataset")
    parser.add_argument('--data_dir', type=str, required=True, help="Path to the compiled training dataset directory")
    parser.add_argument('--out_dir', type=str, default="./training_outputs", help="Directory to saved models and figures")
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--n_folds', type=int, default=5, help="Total number of cross-validation folds")
    parser.add_argument('--save_model', action='store_true', help="Flag to save the best model")
    parser.add_argument('--workers', type=int, default=4, help="Number of CPU workers for data generation")

    return parser.parse_args()

def main():
    args = parse_arguments()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(colored(f"--- KiTS23 Processed Data Model Training ---", "blue", attrs=["bold"]))
    print(f"Dataset Directory: {args.data_dir}")
    print(f"Batch Size: {args.batch_size} | Epochs: {args.epochs} | Total Folds: {args.n_folds}")

    # Determine the input shape from the dataset directory metadata
    metadata_path = Path(args.data_dir) / "dataset_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Could not find dataset metadata at: {metadata_path}")
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    # extract the input shape
    target_shape = metadata["target_shape"]
    input_shape = (target_shape[0], target_shape[1], target_shape[2], 3)
    print(f"Loaded target shape from metadata {metadata_path}")

    print(f"Processing {args.n_folds} folds")
    for fold_no in range(args.n_folds):
        print(f"\n{'='*40}")
        print(colored(f"\tProcessing fold {fold_no}")) 
        print(f"{'='*40}\n")

        # training result output directory
        fold_out_dir = out_dir / f"fold_{fold_no}"
        fold_out_dir.mkdir(exist_ok=True)

        # Get paths
        train_paths, val_paths = dataPrep.get_train_validation_paths(args.data_dir, fold_no)
        # This function splits the dataset into 

        # Extract class weights
        class_weights = dataPrep.get_keras_class_weights(train_paths)

        # Generator
        train_generator = dataPrep.KITSDataGenerator(
            train_paths,
            batch_size=args.batch_size,
            augment=True,
            shuffle=True,
            max_translation=[10,10,10]
        )

        val_generator = dataPrep.KITSDataGenerator(
            val_paths,
            batch_size=args.batch_size,
            augment=False,
            shuffle=False # so that every validation runs identically
        )

        # Build and compile model
        model = mod.build_model(input_shape=input_shape)
        if fold_no == 0:
            model.summary() # print summary only once

        mod.train_model(model, train_generator, val_generator, class_weights, args, fold_no, fold_out_dir)

        plf.plot_and_evaluate(fold_out_dir, fold_no)

        # HPC safety precautions
        del model
        del train_generator
        del val_generator
        keras.backend.clear_session()
        gc.collect()

    plf.plot_and_evaluate_all_folds(args.out_dir)

if __name__ == "__main__":
    main()
