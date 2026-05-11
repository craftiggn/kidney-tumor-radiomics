# NOTE check these values are correct, they are system-specific.
#-------------------------------------------------
dataset_filepath = "/home/marcel/Repos/Data/kits23"
seg_filename = "segmentation.nii.gz"
#-------------------------------------------------

from modules import data_preprocessing as dataPrep
from modules import plotting_functions as plf
import matplotlib.pyplot as plt
import pandas as pd
import os

def list_instance_types(dataset_path, max_cases=589):
    """
    Iterates over all cases and returns a set of unique instance types
    (e.g., 'kidney', 'tumor').
    """
    instance_types = set()

    for case_idx in range(max_cases):
        # print(f"case: {case_idx}")
        case_path = dataPrep.get_case_path(dataset_path, case_idx)
        instances_path = os.path.join(case_path, "instances")

        if not os.path.exists(instances_path):
            # Skip missing cases
            Exception("no instances")
            continue

        for filename in os.listdir(instances_path):
            if not filename.endswith(".nii.gz"):
                continue

            # Example: kidney_instance-1_annotation-2.nii.gz
            if "_instance" in filename:
                instance_type = filename.split("_annotation")[0]
                instance_types.add(instance_type)

    return instance_types


def build_instance_dataframe(dataset_path, instance_types, max_cases=589):
    """
    Returns a pandas DataFrame:
    - rows: case_XXXXX
    - columns: instance types
    - values: True/False (presence of that instance in the case)
    """
    
    # Initialize dictionary to build DataFrame
    data = []

    for case_idx in range(max_cases):
        case_name = f"case_{str(case_idx).zfill(5)}"
        case_path = dataPrep.get_case_path(dataset_path, case_idx)
        instances_path = os.path.join(case_path, "instances")

        # Initialize all False
        case_dict = {inst: False for inst in instance_types}
        case_dict["case"] = case_name

        if os.path.exists(instances_path):
            for filename in os.listdir(instances_path):

                # Skip hidden/system files
                if filename.startswith("."):
                    continue

                if "_annotation" in filename:
                    # Extract "type_instance-X"
                    key = filename.split("_annotation")[0]

                    if key in case_dict:
                        case_dict[key] = True

        data.append(case_dict)

    # Create DataFrame
    df = pd.DataFrame(data)

    # Set case as index (optional but useful)
    # df.set_index("case", inplace=True)

    return df

instances = list_instance_types(dataset_filepath)
print(instances)

df = build_instance_dataframe(dataset_filepath, instances)
print(df.describe())

df_singleTumor_noCyst = df[~df["tumor_instance-2"] & ~df["cyst_instance-1"]]

cases = list(df_singleTumor_noCyst["case"])
print(len(cases) / len(df))

import json
with open("data/single_tumor_cases.json", "w") as f:
    json.dump(cases, f)