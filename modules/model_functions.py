import tensorflow as tf
import keras
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from pathlib import Path
import numpy as np
from termcolor import colored

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

def build_model(input_shape):
    """
    Constructs and compiles the 3D CNN architecture using keras.Sequential.
    """
    model = keras.Sequential([
        keras.layers.Input(shape=input_shape),
        
        keras.layers.Conv3D(32, kernel_size=(3,3,3), activation='relu', padding='same'),
        keras.layers.BatchNormalization(), 
        keras.layers.MaxPooling3D(pool_size=(2,2,2)), 
        
        keras.layers.Conv3D(64, kernel_size=(3, 3, 3), activation='relu', padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling3D(pool_size=(2,2,2)),

        keras.layers.Conv3D(128, kernel_size=(3, 3, 3), activation='relu', padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling3D(pool_size=(2,2,2)),

        # Replaced Flatten() to lower the parameter footprint from 103M to ~30K
        keras.layers.GlobalAveragePooling3D(),

        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dropout(0.4),

        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dropout(0.4),

        keras.layers.Dense(6, activation='sigmoid')
    ], name='RPN_classification_model')

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4), 
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.AUC(multi_label=True)]
    )

    return model

def train_model(model, train_generator, val_generator, class_weights, args, fold_no, fold_out_dir):
    """
    Trains the Sequential model.
    """
    callbacks = [
        CSVLogger(fold_out_dir / f"training_log_fold_{fold_no}.csv", append=False),
        EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True, verbose=1)
    ] 

    if args.save_model:
        model_path = fold_out_dir / f"best_model_fold_{fold_no}.keras"
        checkpoint = ModelCheckpoint(
            filepath=str(model_path),
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
        callbacks.append(checkpoint)

    print(colored(f"Starting training for fold {fold_no}", "blue"))
    
    # Removed workers, use_multiprocessing, and max_queue_size to resolve the TypeError
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=args.epochs,
        class_weight=class_weights,
        callbacks=callbacks
    )

    return history