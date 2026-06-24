import tensorflow as tf
import keras
from pathlib import Path
import numpy as np

def get_keras_class_weights(train_paths):
    """
    Extracts the labels directly from disk and calculates balanced class weights.
    6 output nodes - we calculate the inverse frequency for each node. 
    """
    print("Calculating class weights...")

    all_labels = []
    for path in train_paths:
        # label_path = Path(path) / "labels.npy" # NOTE fix after regenerating the dataset
        label_path = Path(path) / "tabular.npy"
        if label_path.exists():
            all_labels.append(np.load(label_path))

    # Shape (total_samples, 6)
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
    Construcs and compiles the 3D CNN architecture.
    """
    model = keras.Sequential([
        keras.layers.Input(shape=input_shape),
        
        keras.layers.Conv3D(32, kernel_size=(3,3,3), activation='relu', padding='same'),
        keras.layers.BatchNormalization(), # What does this do?
        keras.layers.MaxPooling3D(pool_size=(2,2,2)), # What does this do?
        
        keras.layers.Conv3D(64, kernel_size=(3, 3, 3), activation='relu', padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling3D(pool_size=(2,2,2)),

        keras.layers.Conv3D(128, kernel_size=(3, 3, 3), activation='relu', padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling3D(pool_size=(2,2,2)),

        keras.layers.Flatten(),

        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dropout(0.5),

        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dropout(0.5),

        keras.layers.Dense(6, activation='sigmoid')
    ], name='RPN_classification_model')

    model.summary()

    model.compile(
        optimizer='Adam',
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.AUC(multi_label=True)]
    )

    return model

def train_model(model, train_generator, val_generator, class_weights, args, fold_no, fold_out_dir):
    """
    Trains the model.
    """
    callbacks = 
