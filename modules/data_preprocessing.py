import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt 

def hello():
    print("hello")

def get_instance_numpy(filepath):
    """
    Returns the .nii file as a numpy nd array.
    """
    # https://nipy.org/nibabel/nibabel_images.html

    print(f"getting file from {filepath}.")

    # We load the file to create a nibabel image object:
    img = nib.load(filepath)
    
    # Getting the image data "the easy way"
    image_data = img.get_fdata()
    # <class 'numpy.ndarray'>
    # the data is of schape (slice, x, y)
    print(f"The data is converted into an numpy array of shape {image_data.shape}")
    return image_data

def get_case_path(dataset_filepath: str, instance: int):
    return str(dataset_filepath) + "/case_" +str(instance).zfill(5)

def tumor_size_analisys():
    pass

# Functions for tha labels table

def label_prep():
    pass
    # drop NaN
    # otuput test and training set
    # have constant random kernel


