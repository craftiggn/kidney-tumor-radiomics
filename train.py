import os
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
from datetime import datetime, timedelta
import time
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score, log_loss

# Suppress standard TensorFlow INFO and WARNING logs (set to '3' to mute errors too, '2' filters info/warnings)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# import custom modules
from modules import plotting_functions as plf
from modules import data_preprocessing as dataPrep
from modules import model_functions as mod

def parse_arguments():
    parser = argparse.ArgumentParser(description="Train 3D CNN on KiTS23 Preprocessed Dataset")
    parser.add_argument('--data_dir', type=str, required=True, help="Path to the compiled training dataset directory")
    parser.add_argument('--out_dir', type=str, default="./training_outputs", help="Directory to save models and figures")
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--learning_rate', '--lr', type=float, default=1e-4, help="Learning rate for model optimizer")
    parser.add_argument('--n_folds', type=int, default=5, help="Total number of cross-validation folds")
    parser.add_argument('--save_model', action='store_true', help="Flag to save the best model")
    parser.add_argument('--workers', type=int, default=4, help="Number of CPU workers for data generation")
    parser.add_argument('--architecture', type=str, default='anisotropic_0' ,help="Model architecture key as seen in MODEL_DICT")
    parser.add_argument('--use_lr_scheduler_plateau', action='store_true', help="Enable ReduceLROnPlateau callback")

    return parser.parse_args()

def main():
    start_time = time.time()
    args = parse_arguments()
    config_dict = vars(args).copy()

    # Output Directory setup
    out_dir = Path(args.out_dir)
    timestamp = datetime.now().strftime(r"%Y-%m-%d_%H-%M")
    out_dir = out_dir / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    config_path = out_dir / "run_config.json"

    print(colored(f"--- KiTS23 Processed Data Model Training ---", "blue", attrs=["bold"]))
    print(f"Dataset Directory: {args.data_dir}")
    print(f"Batch Size: {args.batch_size} | Epochs: {args.epochs} | Total Folds: {args.n_folds}")

    # Load input shape
    metadata_path = Path(args.data_dir) / "dataset_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Could not find dataset metadata at: {metadata_path}")
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    target_shape = metadata["target_shape"]
    input_shape = (target_shape[0], target_shape[1], target_shape[2], 3)
    print(f"Loaded target shape from metadata {metadata_path}")

    all_y_true, all_y_pred, fold_summaries = [], [], []
    tabular_cols_ref = None
    model_summary_str = ""

    # --- TRAINING LOOP ---
    for fold_no in range(args.n_folds):
        # --- SETUP ---
        print(f"\n{'='*40}")
        print(colored(f"\tProcessing fold {fold_no}", "blue")) 
        print(f"{'='*40}\n")

        # Fold directory setup
        fold_out_dir = out_dir / f"fold_{fold_no}"
        fold_out_dir.mkdir(exist_ok=True)

        # Dataset setup
        train_paths, val_paths = dataPrep.get_train_validation_paths(args.data_dir, fold_no)
        class_weights = dataPrep.get_keras_class_weights(train_paths)

        # Generator setup
        train_generator = dataPrep.KITSDataGenerator(
            paths=train_paths,
            batch_size=args.batch_size,
            augment=True,
            shuffle=True,
            max_translation=[2, 4, 4],
            workers=args.workers
        )

        val_generator = dataPrep.KITSDataGenerator(
            paths=val_paths,
            batch_size=args.batch_size,
            augment=False,
            shuffle=False
        )

        # Build and compile model
        model = mod.build_model(input_shape=input_shape, model_key=args.architecture)


        # print sumary and save run config
        if fold_no == 0:
            tabular_cols_ref = val_generator.tabular_cols
            summary_lines = []
            model.summary(print_fn=lambda x: summary_lines.append(x))
            model_summary_str = "\n".join(summary_lines)
            print(model_summary_str)

            plf.plot_model_architecture(model, out_dir / "model_architecture.png")

            # Store complete run configuration with all metadata
            config_dict["input_shape"] = list(input_shape)
            config_dict["keras_model_name"] = model.name
            config_dict["total_parameters"] = int(model.count_params())
            config_dict["run_timestamp"] = timestamp

            with open(config_path, 'w') as f:
                json.dump(config_dict, f, indent=4)
            print(colored(f"Saved run configuration to {config_path}", "green"))

        # --- TRAINING ---
        if args.use_lr_scheduler_plateau:
            lr_scheduler = keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.8,
                patience=10,
                min_lr=1e-6,
                verbose=0
            )
        else:
            lr_scheduler = None

        history, y_true, y_pred = mod.train_model(
            model=model, 
            train_generator=train_generator, 
            val_generator=val_generator, 
            class_weights=class_weights, 
            args=args, 
            fold_no=fold_no, 
            fold_out_dir=fold_out_dir, 
            learning_rate=args.learning_rate,
            lr_scheduler=lr_scheduler
        )

        # --- FOLD EVALUATION ---
        all_y_true.append(y_true)
        all_y_pred.append(y_pred)

        metrics = mod.evaluate_and_save_fold(y_true, y_pred, val_generator.tabular_cols, fold_no, fold_out_dir)
        fold_summaries.append(metrics)
        plf.plot_and_evaluate(fold_out_dir, fold_no, val_generator.tabular_cols)

        # HPC Cleanup
        del model
        del train_generator
        del val_generator
        keras.backend.clear_session()
        gc.collect()

    # --- OVERALL MODEL EVALUATION ---
    plf.plot_and_evaluate_all_folds(out_dir)
    plf.plot_overall_confusion_matrix(all_y_true, all_y_pred, tabular_cols_ref, out_dir)
    mod.save_overall_summary(fold_summaries, model_summary_str, out_dir)

    # add the run length information to the run_config.json
    total_seconds = int(time.time() - start_time)
    formatted_time = str(timedelta(seconds=int(total_seconds)))
    seconds_per_fold = total_seconds / args.n_folds
    avg_fold_time_formated = str(timedelta(seconds=int(seconds_per_fold)))

    config_dict["total_runtime_seconds"] = total_seconds
    config_dict["total_runtime_formatted"] = formatted_time
    config_dict["average_fold_runtime_formatted"] = avg_fold_time_formated

    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=4)

    print(colored(f"Training time completed in {formatted_time} ({total_seconds}s)"))

if __name__ == "__main__":
    main()