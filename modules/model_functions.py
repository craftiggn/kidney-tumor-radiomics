import tensorflow as tf
import keras
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from pathlib import Path
import numpy as np
import pandas as pd
from termcolor import colored
from modules import plotting_functions as plf
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, log_loss
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
    ], name='RPN_classification_model_anisotropic_regularized_0') 

def model_anisotropic_reg_1(input_shape):
    """3D CNN anisotropic model, with added regularization, and deeper filters."""
    reg = keras.regularizers.L2(1e-4)

    return keras.Sequential([
        keras.layers.Input(shape=input_shape),

        keras.layers.Conv3D(32, kernel_size=(1, 3, 3), activation='relu', padding='same', kernel_regularizer=reg),
        keras.layers.BatchNormalization(),
        keras.layers.SpatialDropout3D(0.3),
        keras.layers.MaxPooling3D(pool_size=(1, 2, 2)),

        keras.layers.Conv3D(64, kernel_size=(3, 3, 3), use_bias=False, padding='same', kernel_regularizer=reg),
        keras.layers.BatchNormalization(),
        keras.layers.Activation('relu'),
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
    ], name='RPN_classification_model_anisotropic_regularized_1')

def model_anisotropic_reg_2(input_shape):
    """3D CNN anisotropic model, with added regularization, and more dense layers."""
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

        keras.layers.Conv3D(256, kernel_size=(3, 3, 3), activation='relu', padding='same', kernel_regularizer=reg),
        keras.layers.BatchNormalization(),
        keras.layers.SpatialDropout3D(0.3),
        keras.layers.MaxPooling3D(pool_size=(2, 2, 2)),

        keras.layers.GlobalAveragePooling3D(),

        keras.layers.Dense(256, activation='relu', kernel_regularizer=reg),
        keras.layers.Dropout(0.3),

        keras.layers.Dense(128, activation='relu', kernel_regularizer=reg),
        keras.layers.Dropout(0.2),

        keras.layers.Dense(64, activation='relu', kernel_regularizer=reg),
        keras.layers.Dropout(0.1),

        keras.layers.Dense(32, activation='relu', kernel_regularizer=reg),

        keras.layers.Dense(6, activation='sigmoid')
    ], name='RPN_classification_model_anisotropic_regularized_2') 

def model_anisotropic_reg_2_1(input_shape):
    """3D CNN anisotropic model, double kernel"""
    reg = keras.regularizers.L2(1e-4)

    return keras.Sequential([
        keras.layers.Input(shape=input_shape),

        keras.layers.Conv3D(32, kernel_size=(1, 3, 3), activation='relu', padding='same', kernel_regularizer=reg),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling3D(pool_size=(1, 2, 2)),

        keras.layers.Conv3D(64, kernel_size=(3, 3, 3), use_bias=False, padding='same', kernel_regularizer=reg),
        keras.layers.BatchNormalization(),
        keras.layers.Activation('relu'),
        keras.layers.Conv3D(64, kernel_size=(3, 3, 3), activation='relu', padding='same', kernel_regularizer=reg),
        keras.layers.BatchNormalization(),
        keras.layers.SpatialDropout3D(0.2),
        keras.layers.MaxPooling3D(pool_size=(2, 2, 2)),

        keras.layers.Conv3D(128, kernel_size=(3, 3, 3), activation='relu', padding='same', kernel_regularizer=reg),
        keras.layers.BatchNormalization(),
        keras.layers.SpatialDropout3D(0.3),
        keras.layers.MaxPooling3D(pool_size=(2, 2, 2)),

        keras.layers.Conv3D(256, kernel_size=(3, 3, 3), activation='relu', padding='same', kernel_regularizer=reg),
        keras.layers.BatchNormalization(),
        keras.layers.SpatialDropout3D(0.3),
        keras.layers.MaxPooling3D(pool_size=(2, 2, 2)),

        keras.layers.GlobalAveragePooling3D(),

        keras.layers.Dense(256, activation='relu', kernel_regularizer=reg),
        keras.layers.Dropout(0.3),

        keras.layers.Dense(128, activation='relu', kernel_regularizer=reg),
        keras.layers.Dropout(0.2),

        keras.layers.Dense(64, activation='relu', kernel_regularizer=reg),
        keras.layers.Dropout(0.1),

        keras.layers.Dense(32, activation='relu', kernel_regularizer=reg),

        keras.layers.Dense(6, activation='sigmoid')
    ], name='RPN_classification_model_anisotropic_regularized_2_1') 

def model_anisotropic_reg_2_2(input_shape):
    """3D CNN anisotropic model, wider and deeper dense layers"""
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

        keras.layers.Conv3D(256, kernel_size=(3, 3, 3), activation='relu', padding='same', kernel_regularizer=reg),
        keras.layers.BatchNormalization(),
        keras.layers.SpatialDropout3D(0.3),
        keras.layers.MaxPooling3D(pool_size=(2, 2, 2)),

        keras.layers.GlobalAveragePooling3D(),

        keras.layers.Dense(512, activation='relu', kernel_regularizer=reg),
        keras.layers.Dropout(0.2),

        keras.layers.Dense(512, activation='relu', kernel_regularizer=reg),
        keras.layers.Dropout(0.2),

        keras.layers.Dense(256, activation='relu', kernel_regularizer=reg),
        keras.layers.Dropout(0.2),

        keras.layers.Dense(128, activation='relu', kernel_regularizer=reg),
        keras.layers.Dropout(0.2),

        keras.layers.Dense(32, activation='relu', kernel_regularizer=reg),

        keras.layers.Dense(6, activation='sigmoid')
    ], name='RPN_classification_model_anisotropic_regularized_2_2') 

def model_anisotropic_reg_2_3(input_shape):
    """3D CNN anisotropic model, fewer Dense layers and more filters"""
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

        keras.layers.Conv3D(256, kernel_size=(3, 3, 3), activation='relu', padding='same', kernel_regularizer=reg),
        keras.layers.BatchNormalization(),
        keras.layers.SpatialDropout3D(0.3),
        keras.layers.MaxPooling3D(pool_size=(2, 2, 2)),

        keras.layers.Conv3D(512, kernel_size=(3, 3, 3), activation='relu', padding='same', kernel_regularizer=reg),
        keras.layers.BatchNormalization(),
        keras.layers.SpatialDropout3D(0.3),
        keras.layers.MaxPooling3D(pool_size=(2, 2, 2)),

        keras.layers.GlobalAveragePooling3D(),

        keras.layers.Dense(128, activation='relu', kernel_regularizer=reg),
        keras.layers.Dropout(0.2),

        keras.layers.Dense(64, activation='relu', kernel_regularizer=reg),
        keras.layers.Dropout(0.1),

        keras.layers.Dense(32, activation='relu', kernel_regularizer=reg),

        keras.layers.Dense(6, activation='sigmoid')
    ], name='RPN_classification_model_anisotropic_regularized_2_3') 

def model_anisotropic_reg_3(input_shape):
    """3D CNN with Dual Pooling, Swish, and Learned Downsampling"""
    reg = keras.regularizers.L2(1e-4)
    inputs = keras.layers.Input(shape=input_shape)

    # --- BLOCK 1: Anisotropic ---
    x = keras.layers.Conv3D(32, kernel_size=(1, 3, 3), activation='swish', padding='same', kernel_regularizer=reg)(inputs)
    x = keras.layers.BatchNormalization()(x)
    # Learned downsampling (strides instead of pool)
    x = keras.layers.Conv3D(32, kernel_size=(1, 3, 3), strides=(1, 2, 2), activation='swish', padding='same')(x)

    # --- BLOCK 2: Isotropic ---
    x = keras.layers.Conv3D(64, kernel_size=(3, 3, 3), activation='swish', padding='same', kernel_regularizer=reg)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.SpatialDropout3D(0.2)(x)
    x = keras.layers.Conv3D(64, kernel_size=(3, 3, 3), strides=(2, 2, 2), activation='swish', padding='same')(x)

    # --- BLOCK 3 ---
    x = keras.layers.Conv3D(128, kernel_size=(3, 3, 3), activation='swish', padding='same', kernel_regularizer=reg)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.SpatialDropout3D(0.3)(x)
    x = keras.layers.Conv3D(128, kernel_size=(3, 3, 3), strides=(2, 2, 2), activation='swish', padding='same')(x)

    # --- BLOCK 4 ---
    x = keras.layers.Conv3D(256, kernel_size=(3, 3, 3), activation='swish', padding='same', kernel_regularizer=reg)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.SpatialDropout3D(0.3)(x)
    x = keras.layers.Conv3D(256, kernel_size=(3, 3, 3), strides=(2, 2, 2), activation='swish', padding='same')(x)

    # --- DUAL POOLING HEAD ---
    avg_pool = keras.layers.GlobalAveragePooling3D()(x)
    max_pool = keras.layers.GlobalMaxPooling3D()(x)
    x = keras.layers.Concatenate()([avg_pool, max_pool]) # Combines both representations

    # --- DENSE LAYERS ---
    x = keras.layers.Dense(256, activation='swish', kernel_regularizer=reg)(x)
    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.Dense(128, activation='swish', kernel_regularizer=reg)(x)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Dense(64, activation='swish', kernel_regularizer=reg)(x)
    x = keras.layers.Dropout(0.1)(x)
    
    outputs = keras.layers.Dense(6, activation='sigmoid')(x)

    return keras.Model(inputs=inputs, outputs=outputs, name='RPN_model_3')

MODEL_DICT = {
    'anisotropic_0': model_anisotropic_0,
    'anisotropic_reg_0': model_anisotropic_reg_0,
    'anisotropic_reg_1': model_anisotropic_reg_1,
    'anisotropic_reg_2': model_anisotropic_reg_2,
    'anisotropic_reg_2_1': model_anisotropic_reg_2_1,
    'anisotropic_reg_2_2': model_anisotropic_reg_2_2,
    'anisotropic_reg_2_3': model_anisotropic_reg_2_3,
    'anisotropic_reg_3': model_anisotropic_reg_3,
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
    Trains the Sequential model and returns history along with true and predicted labels.
    """
    log_csv_path = fold_out_dir / f"training_log_fold_{fold_no}.csv"
    
    # Custom logger callback replacing default CSVLogger
    metrics_logger = PerClassMetricsCallback(
        train_gen=train_generator,
        val_gen=val_generator,
        class_names=val_generator.tabular_cols,
        log_path=log_csv_path
    )
    
    callbacks = [metrics_logger]

    if isinstance(learning_rate, keras.callbacks.Callback):
        callbacks.append(learning_rate)
        optimizer = keras.optimizers.Adam()
    else:
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    # for custom schedulers
    if lr_scheduler is not None:
        callbacks.append(lr_scheduler)

    early_stopping_patience = 64
    callbacks.append(EarlyStopping(monitor='val_loss', patience=early_stopping_patience, restore_best_weights=True, verbose=1))
    
    if args.save_model:
        model_path = fold_out_dir / f"best_model_fold_{fold_no}.keras"
        callbacks.append(ModelCheckpoint(filepath=str(model_path), monitor='val_loss', save_best_only=True, verbose=1))

    print(colored(f"Starting training for fold {fold_no}", "green"))
    
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=args.epochs,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=0
    )

    # final validation predictions
    y_true_list, y_pred_list = [], []
    for i in range(len(val_generator)):
        X_batch, y_batch = val_generator[i]
        preds = model.predict(X_batch, verbose=0)
        y_true_list.append(y_batch)
        y_pred_list.append(preds)

    y_true = np.concatenate(y_true_list, axis=0)
    y_pred = np.concatenate(y_pred_list, axis=0)

    # fold plots
    plf.plot_multilabel_roc_and_pr_curves(
        y_true=y_true, y_pred=y_pred, fold_no=fold_no, out_dir=fold_out_dir, class_names=val_generator.tabular_cols
    )
    plf.plot_feature_confusion_matrix(
        y_true=y_true, y_pred=y_pred, fold_no=fold_no, out_dir=fold_out_dir, tabula_cols=val_generator.tabular_cols
    )

    return history, y_true, y_pred



class PerClassMetricsCallback(keras.callbacks.Callback):
    """
    Evaluates and records per-label: Loss, AUC, AP, F1 score on
    train and val datasets at the end of each epoch, saving the values to a .csv file.
    """

    def __init__(self, train_gen, val_gen, class_names, log_path):
        super().__init__()
        self.train_gen = train_gen
        self.val_gen = val_gen

        self.class_names = class_names
        self.log_path = Path(log_path)

        self.history_data = []

    def on_epoch_end(self, epoch, logs=None):
        # Validation predictions
        y_val_true_list, y_val_pred_list = [], []
        for i in range(len(self.val_gen)):
            X_b, y_b = self.val_gen[i]
            preds = self.model.predict(X_b, verbose=0)
            y_val_true_list.append(y_b)
            y_val_pred_list.append(preds)

        y_val_true = np.concatenate(y_val_true_list, axis=0)
        y_val_pred = np.concatenate(y_val_pred_list, axis=0)

        # Training predictions
        y_tr_true_list, y_tr_pred_list = [], []
        for i in range(len(self.train_gen)):
            X_b, y_b = self.train_gen[i]
            preds = self.model.predict(X_b, verbose=0)
            y_tr_true_list.append(y_b)
            y_tr_pred_list.append(preds)

        y_tr_true = np.concatenate(y_tr_true_list, axis=0)
        y_tr_pred = np.concatenate(y_tr_pred_list, axis=0)

        row = {'epoch': epoch}
        tr_losses, val_losses, tr_aucs, val_aucs = [], [], [], []
        tr_aps, val_aps, tr_f1s, val_f1s = [], [], [], []

        for i, name in enumerate(self.class_names):
            # Loss
            tr_l = log_loss(y_tr_true[:, i], y_tr_pred[:, i], labels=[0, 1])
            val_l = log_loss(y_val_true[:, i], y_val_pred[:, i], labels=[0, 1])

            # ROC AUC
            try: tr_auc = roc_auc_score(y_tr_true[:, i], y_tr_pred[:, i])
            except ValueError: tr_auc = 0.5
            try: val_auc = roc_auc_score(y_val_true[:, i], y_val_pred[:, i])
            except ValueError: val_auc = 0.5

            # AP
            tr_ap = average_precision_score(y_tr_true[:, i], y_tr_pred[:, i])
            val_ap = average_precision_score(y_val_true[:, i], y_val_pred[:, i])

            # F1 
            tr_f1 = f1_score(y_tr_true[:, i], (y_tr_pred[:, i] > 0.5).astype(int), zero_division=0)
            val_f1 = f1_score(y_val_true[:, i], (y_val_pred[:, i] > 0.5).astype(int), zero_division=0)

            row[f'train_loss_{name}'] = tr_l
            row[f'val_loss_{name}'] = val_l
            row[f'train_auc_{name}'] = tr_auc
            row[f'val_auc_{name}'] = val_auc
            row[f'train_ap_{name}'] = tr_ap
            row[f'val_ap_{name}'] = val_ap
            row[f'train_f1_{name}'] = tr_f1
            row[f'val_f1_{name}'] = val_f1

            tr_losses.append(tr_l)
            val_losses.append(val_l)
            tr_aucs.append(tr_auc)
            val_aucs.append(val_auc)
            tr_aps.append(tr_ap)
            val_aps.append(val_ap)
            tr_f1s.append(tr_f1)
            val_f1s.append(val_f1)

        # averages
        row['loss'] = float(np.mean(tr_losses))
        row['val_loss'] = float(np.mean(val_losses))
        row['auc'] = float(np.mean(tr_aucs))
        row['val_auc'] = float(np.mean(val_aucs))
        row['ap'] = float(np.mean(tr_aps))
        row['val_ap'] = float(np.mean(val_aps))
        row['f1'] = float(np.mean(tr_f1s))
        row['val_f1'] = float(np.mean(val_f1s))

        self.history_data.append(row)
        pd.DataFrame(self.history_data).to_csv(self.log_path, index=False)



def evaluate_and_save_fold(y_true, y_pred, tabular_cols, fold_no, out_dir):
    """
    Calculates fold metrics and saves a text summary.
    """
    y_pred_bin = (y_pred > 0.5).astype(int)

    metrics = {
        'fold': fold_no,
        'f1': f1_score(y_true, y_pred_bin, average='macro', zero_division=0),
        'auc': roc_auc_score(y_true, y_pred, average='macro'),
        'ap': average_precision_score(y_true, y_pred, average='macro'),
        'loss': log_loss(y_true, y_pred)
    }

    # per-label string
    label_evals = []
    for i, col in enumerate(tabular_cols):
        label_evals.append(f"{col:20s} | F1: {f1_score(y_true[:, i], y_pred_bin[:, i], zero_division=0):.4f} | AP: {average_precision_score(y_true[:, i], y_pred[:, i]):.4f}")

    summary_text = f"""--- FOLD {fold_no} EVALUATION SUMMARY ---
Macro F1 Score  : {metrics['f1']:.4f}
Macro ROC AUC   : {metrics['auc']:.4f}
Macro AP        : {metrics['ap']:.4f}
Validation Loss : {metrics['loss']:.4f}

--- PER-LABEL EVALUATION ---
{chr(10).join(label_evals)}
"""
    eval_path = out_dir / f"metrics_summary_fold{fold_no}.txt" 
    eval_path.write_text(summary_text)
    print(colored(f"Fold evaluation saved to: {eval_path}", "green"))
    return metrics



def save_overall_summary(fold_summaries, model_summary_str, out_dir):
    """Aggregates all fold metrics and generates the final model report."""
    df = pd.DataFrame(fold_summaries)
    best_fold = df.loc[df['f1'].idxmax()]
    worst_fold = df.loc[df['f1'].idxmin()]

    fold_details = [
        f"Fold {int(row.fold)} | F1: {row.f1:.4f} | AUC: {row.auc:.4f} | AP: {row.ap:.4f}"
        for _, row in df.iterrows()
    ]

    report = f"""--- MODEL SUMMARY ---
Mean F1 Score       : {df['f1'].mean():.4f} +/- {df['f1'].std():.4f}
Mean ROC AUC        : {df['auc'].mean():.4f} +/- {df['auc'].std():.4f}
Mean AP             : {df['ap'].mean():.4f} +/- {df['ap'].std():.4f}

Best Fold F1 Score  : {best_fold.f1:.4f} (Fold {int(best_fold.fold)})
Worst Fold F1 Score : {worst_fold.f1:.4f} (Fold {int(worst_fold.fold)})

--- PER-FOLD SUMMARY ---
{chr(10).join(fold_details)}

--- MODEL ARCHITECTURE ---
{model_summary_str}
"""
    report_path = out_dir / "model_evaluation_summary.txt"
    report_path.write_text(report)
    print(colored(f"Model evaluation report saved to {report_path}", "green"))