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
from datetime import datetime, timedelta
import time
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score, log_loss

# import custom modules
from modules import plotting_functions as plf
from modules import data_preprocessing as dataPrep
from modules import model_functions as mod

# sample run command:
# python3 train.py --data_dir /home/marcel/Repos/Data/kits23_processed/ --epochs 10 --n_folds 1 --save_model --architecture anisotropic_0
# 

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

    out_dir = Path(args.out_dir)
    timestamp = datetime.now().strftime(r"%Y-%m-%d_%H-%M") # to help keep trac of trainings
    out_dir = out_dir / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    config_path = out_dir / "run_config.json"

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

    all_y_true, all_y_pred, fold_summaries = [], [], []
    tabular_cols_ref = None
    model_summary_str = ""

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

        # --- Generators ---
        train_generator = dataPrep.KITSDataGenerator(
            train_paths,
            batch_size=args.batch_size,
            augment=True,
            shuffle=True,
            max_translation=[2, 4, 4],
            workers=args.workers
        )

        val_generator = dataPrep.KITSDataGenerator(
            val_paths,
            batch_size=args.batch_size,
            augment=False,
            shuffle=False # so that every validation runs identically
        )

        # --- Build and compile model ---
        model = mod.build_model(input_shape=input_shape, model_key=args.architecture)

        # print sumary and save run config
        if fold_no == 0:
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

        # Instantiate LR Scheduler if flagged in arguments
        if args.use_lr_scheduler_plateau:
            lr_scheduler = keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.8,
                patience=10,
                min_lr=1e-6,
                verbose=1
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

        all_y_true.append(y_true)
        all_y_pred.append(y_pred)

        if tabular_cols_ref is None:
            tabular_cols_ref = val_generator.tabular_cols

        # process fold numerical scores
        y_pred_bin = (y_pred > 0.5).astype(int)
        fold_f1 = f1_score(y_true, y_pred_bin, average='macro', zero_division=0)
        fold_auc = roc_auc_score(y_true, y_pred, average='macro')
        fold_ap = average_precision_score(y_true, y_pred, average='macro')
        fold_loss = log_loss(y_true, y_pred)

        fold_metrics = {
            'fold': fold_no,
            'f1': fold_f1,
            'auc': fold_auc,
            'ap': fold_ap,
            'loss': fold_loss
        }
        fold_summaries.append(fold_metrics)

        # Per fold performance summary
        fold_txt_path = fold_out_dir / f"metrics_summary_fold{fold_no}.txt"
        with open(fold_txt_path, 'w') as f:
            f.write(f"--- FOLD {fold_no} EVALUATION SUMMARY ---\n")
            f.write(f"Macro F1 Score  : {fold_f1:.4f}\n")
            f.write(f"Macro ROC AUC   : {fold_auc:.4f}\n")
            f.write(f"Macro AP        : {fold_ap:.4f}\n")
            f.write(f"Validation Loss : {fold_loss:.4f}\n\n")

            f.write("--- PER-LABEL EVALUATION ---\n")
            for i, col in enumerate(val_generator.tabular_cols):
                lbl_f1 = f1_score(y_true[:, i], y_pred_bin[:, i], zero_division=0)
                lbl_ap = average_precision_score(y_true[:, i], y_pred[:, i])
                f.write(f"{col:20s} | F1: {lbl_f1:.4f} | AP: {lbl_ap:.4f}\n")

        plf.plot_and_evaluate(fold_out_dir, fold_no, val_generator.tabular_cols)

        # HPC safety precautions
        del model
        del train_generator
        del val_generator
        keras.backend.clear_session()
        gc.collect()

    # --- MODEL EVALUATION ACROSS ALL FOLDS ---
    plf.plot_and_evaluate_all_folds(out_dir)
    plf.plot_overall_confusion_matrix(all_y_true, all_y_pred, tabular_cols_ref, out_dir)

    f1s = [m['f1'] for m in fold_summaries]
    aucs = [m['auc'] for m in fold_summaries]
    aps = [m['ap'] for m in fold_summaries]
    losses = [m['loss'] for m in fold_summaries]

    best_fold_idx = int(np.argmax(f1s))
    worst_fold_idx = int(np.argmin(f1s))

    overall_txt_path = out_dir / "model_evaluation_summary.txt"
    with open(overall_txt_path, 'w') as f:
        f.write(f"--- MODEL SUMMARY ---\n")
        f.write(f"Mean F1 Score       : {np.mean(f1s):.4f} +/- {np.std(f1s):.4f}\n")
        f.write(f"Mean ROC AUC        : {np.mean(aucs):.4f} +/- {np.std(aucs):.4f}\n")
        f.write(f"Mean AP             : {np.mean(aps):.4f} +/- {np.std(aps):.4f}\n\n")
        f.write(f"Best Fold F1 Score  : {f1s[best_fold_idx]:.4f} (Fold {best_fold_idx})\n")
        f.write(f"Worst Fold F1 Score : {f1s[worst_fold_idx]:.4f} (Fold {worst_fold_idx})\n")

        f.write(f"--- PER-FOLD SUMMARY ---\n")
        for m in fold_summaries:
            f.write(f"Fold {m['fold']} | F1 {m['f1']:.4f} | AUC: {m['auc']:.4f} | AP: {m['ap']:.4f}\n")

        f.write("--- MODEL ARCHITECTURE ---\n")
        f.write(model_summary_str)

    print(colored(f"Model evaluation report saved to {overall_txt_path}", "green"))


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
