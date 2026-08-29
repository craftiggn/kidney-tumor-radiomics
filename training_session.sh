#!/bin/bash -l
# run with: sbatch training_session.sh

# --- SLURM DIRECTIVES (Resource Requests) ---

#SBATCH --job-name=training_session
#SBATCH --account=g105-2708
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=07:00:00
#SBATCH --output=training_session_%j.log

# --- SCRIPT COMMANDS ---

# print start information on the log file
echo "Job started on $(hostname) at $(date)"

# cd into the project folder
cd /home/marcel42/kidney-tumor-radiomics

# custom installed libraries
export PYTHONUSERBASE=/home/marcel42/kidney-tumor-radiomics/env_libs
export PATH=$PYTHONUSERBASE/bin:$PATH

# install libraries
apptainer exec --nv /lu/tetyda/projects/center4ml/Containers/tensorflow-gpu.sif pip3 install --user -r requirements.txt

# Run python script within container (with the libraries)
apptainer exec --nv /lu/tetyda/projects/center4ml/Containers/tensorflow-gpu.sif python3 train.py --data_dir /home/marcel42/Data/kits23_processed/kits_dataset_36_60_60_kidney --out_dir /home/marcel42/kidney-tumor-radiomics/HPC_out --epochs 200 --n_folds 3 --workers 8 --architecture model_anisotropic_reg_2_1  --use_lr_scheduler_plateau

# apptainer exec --nv /lu/tetyda/projects/center4ml/Containers/tensorflow-gpu.sif python3 train.py --data_dir /home/marcel42/Data/kits23_processed/kits_dataset_36_60_60_kidney --out_dir /home/marcel42/kidney-tumor-radiomics/HPC_out --epochs 200 --n_folds 3 --workers 8 --architecture model_anisotropic_reg_2_2  --use_lr_scheduler_plateau

# apptainer exec --nv /lu/tetyda/projects/center4ml/Containers/tensorflow-gpu.sif python3 train.py --data_dir /home/marcel42/Data/kits23_processed/kits_dataset_36_60_60_kidney --out_dir /home/marcel42/kidney-tumor-radiomics/HPC_out --epochs 200 --n_folds 3 --workers 8 --architecture model_anisotropic_reg_2_3  --use_lr_scheduler_plateau

# apptainer exec --nv /lu/tetyda/projects/center4ml/Containers/tensorflow-gpu.sif python3 train.py --data_dir /home/marcel42/Data/kits23_processed/kits_dataset_36_60_60_kidney --out_dir /home/marcel42/kidney-tumor-radiomics/HPC_out --epochs 200 --n_folds 3 --workers 8 --architecture model_anisotropic_reg_3  --use_lr_scheduler_plateau

# Print completion status
echo "Job finished at $(date)"