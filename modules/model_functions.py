import tensorflow as tf
import keras
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from pathlib import Path
import numpy as np
from termcolor import colored
from modules import plotting_functions as plf
# from keras.regularizers import L2

# --- MODEL DEFINITIONS ---

def model_anisotropic_0(input_shape):
    """Initial 3D CNN anisotropic model."""
    return keras.Sequential([
        keras.layers.Input(shape=input_shape),

        # --- LAYER 1: Anisotropic Processing ---
        # kernel_size=(1,3,3) forces the network to learn 2D spatial features first
        # pool_size=(1,2,2) shrinks X/Y but preserves the shallow Z-axis resolution
        keras.layers.Conv3D(32, kernel_size=(1, 3, 3), activation='relu', padding='same'),
        keras.layers.BatchNormalization(), 
        keras.layers.MaxPooling3D(pool_size=(1, 2, 2)), 

        # --- LAYER 2: Transition to Isotropic ---
        # Now that X/Y are halved, the feature map proportions are closer to cubic.
        # We can start aggregating 3D context safely.
        keras.layers.Conv3D(64, kernel_size=(3, 3, 3), activation='relu', padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling3D(pool_size=(2, 2, 2)),

        # --- LAYER 3: Fully Isotropic ---
        keras.layers.Conv3D(128, kernel_size=(3, 3, 3), activation='relu', padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling3D(pool_size=(2, 2, 2)),

        # Global Average Pooling flattens the remaining spatial dimensions beautifully
        keras.layers.GlobalAveragePooling3D(),

        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dropout(0.4),

        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dropout(0.4),

        keras.layers.Dense(6, activation='sigmoid')
        ], name='RPN_classification_model_anisotropic')

def model_anisotropic_reg_0(input_shape):
    """3D CNN anisotropic model, with added regularization"""
    reg = keras.regularizers.L2(1e-4)

    return keras.Sequential([
        keras.layers.Input(shape=input_shape),

        keras.layers.Conv3D(32, kernel_size=(1, 3, 3), activation='relu', padding='same', kernel_regularizer=reg),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling3D(pool_size=(1, 2, 2)),

        keras.layers.Conv3D(64, kernel_size=(3, 3, 3), activation='relu', padding='same', kernel_regularizer=reg),
        keras.layers.BatchNormalization(),
        keras.layers.SpatialDropout3D(0.2),
        keras.layers.MaxPooling3D(pool_size=(2, 2, 2)),

        keras.layers.Conv3D(128, kernel_size=(3, 3, 3), activation='relu', padding='same', kernel_regularizer=reg),
        keras.layers.BatchNormalization(),
        keras.layers.SpatialDropout3D(0.3),
        keras.layers.MaxPooling3D(pool_size=(2, 2, 2)),

        keras.layers.GlobalAveragePooling3D(),

        keras.layers.Dense(256, activation='relu', kernel_regularizer=reg),
        keras.layers.Dropout(0.5),

        keras.layers.Dense(6, activation='sigmoid')
    ], name='RPN_classification_model_anisotropic_regularized') 

MODEL_DICT = {
    'anisotropic_0': model_anisotropic_0,
    'anisotropic_reg_0': model_anisotropic_reg_0,
}

# --- FUNCTIONS ---

def get_keras_class_weights(train_paths):
    """
    Extracts the labels directly from disk and calculates balanced class weights.
    6 output nodes - we calculate the inverse frequency for each node. 
    """
    print("Calculating class weights...")
    all_labels = []
    for path in train_paths:
        label_path = Path(path) / "tabular.npy"
        if label_path.exists():
            all_labels.append(np.load(label_path))

    labels_matrix = np.stack(all_labels)
    total_samples = labels_matrix.shape[0]
    
    class_weights = {}
    for i in range(labels_matrix.shape[1]):
        positives = np.sum(labels_matrix[:, i])
        weight = total_samples / (2.0 * positives) if positives > 0 else 1.0
        class_weights[i] = weight

    print(f"Calculated node weights: {class_weights}")
    return class_weights

def build_model(input_shape, model_key):
    """
    Constructs and compiles the 3D CNN architecture using keras.Sequential, based on a given key.
    """

    if model_key not in MODEL_DICT:
        raise ValueError(colored(f"Model key '{model_key}' not found in MODEL_DICT.\nAvailable keys:{list(MODEL_DICT.keys())}", "red"))

    print(f"Building model using architecture: {model_key}")
    model = MODEL_DICT[model_key](input_shape)

    return model

def train_model(model, train_generator, val_generator, class_weights, args, fold_no, fold_out_dir, learning_rate=1e-4, lr_scheduler=None):
    """
    Trains the Sequential model.
    """
    callbacks = [
        CSVLogger(fold_out_dir / f"training_log_fold_{fold_no}.csv", append=False)
    ]

    # Dynamic vs Static Learning
    if isinstance(learning_rate, keras.callbacks.Callback):
        # callbacks like ReduceLROnPlateau, append it
        print(colored("Using dynamic learning rate via callback.", "blue"))
        callbacks.append(learning_rate)
        optimizer = keras.optimizers.Adam()
    elif isinstance(learning_rate, float):
        print(colored(f"Using static learning rate: {learning_rate}", "blue"))
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    else:
        print(colored("Using dynamic learning rate via Schedule.", "blue"))
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

    # compile model before training
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.AUC(multi_label=True)]
    )

    # --- LR Scheduler ---
    if lr_scheduler is not None:
        callbacks.append(lr_scheduler)
        print(colored("LR Scheduler ENABLED via parameter.", "blue"))
    else:
        print(colored(f"LR Scheduler DISABLED. Using static LR: {learning_rate}", "blue"))
    
    # --- Early Stopping ---
    early_stopping_patience = 64
    callbacks.append(EarlyStopping(monitor='val_loss', patience=early_stopping_patience, restore_best_weights=True, verbose=1))
    print(colored(f"Early stopping ENABLED (hardcoded patience={early_stopping_patience}).", "blue"))
    
    if args.save_model:
        model_path = fold_out_dir / f"best_model_fold_{fold_no}.keras"
        checkpoint = ModelCheckpoint(
            filepath=str(model_path),
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
        callbacks.append(checkpoint)

    print(colored(f"Starting training for fold {fold_no}", "green"))
    
    # --- Model Training ---
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=args.epochs,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=0
    )

    # --- ROC and PR curves --- 
    # Extract true labels from the val_generator
    print(colored("Generating validation predictions for ROC and PR curves...", "blue"))

    y_true_list = []
    y_pred_list = []

    # Iterate through the val dataset
    for i in range(len(val_generator)):
        X_batch, y_batch = val_generator[i]
        preds = model.predict(X_batch, verbose=0)

        y_true_list.append(y_batch)
        y_pred_list.append(preds)

    # Concat into arrays
    y_true = np.concatenate(y_true_list, axis=0)
    y_pred = np.concatenate(y_pred_list, axis=0)

    # plot ROC, PR
    plf.plot_multilabel_roc_and_pr_curves(
        y_true=y_true,
        y_pred=y_pred,
        fold_no=fold_no,
        out_dir=fold_out_dir,
        class_names=val_generator.tabular_cols
    )

    plf.plot_feature_confusion_matrix(
        y_true=y_true,
        y_pred=y_pred,
        fold_no=fold_no,
        out_dir=fold_out_dir,
        tabula_cols=val_generator.tabular_cols
    )

    return history