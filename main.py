#NOTE check these values are correct, they are system-specific.
#-------------------------------------------------
dataset_filepath = "/home/marcel/Repos/Data/kits23"
#-------------------------------------------------

import sys
sys.path.append("./modules")

from modules import data_preprocessing as dataPrep
from modules import plotting_functions as plf
import matplotlib.pyplot as plt


instance_path = "/home/marcel/Repos/Data/kits23/case_00000/"
img_filename = "imaging.nii.gz"
seg_filename = "segmentation.nii.gz"

data_seg = dataPrep.get_instance_numpy(instance_path+seg_filename)
# data_img = dataPrep.get_instance_numpy(instance_path+img_filename)




plf.show_value_counts(data_seg)
