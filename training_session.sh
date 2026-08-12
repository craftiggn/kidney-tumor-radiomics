#!/bin/bash -l
# run with: sbatch training_session.sh

# --- SLURM DIRECTIVES (Resource Requests) ---

#SBATCH --job-name=training_session
#SBATCH --account=g105-2708
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=training_session_%j.log

# --- SCRIPT COMMANDS ---

# print start information on the log file
echo "Job started on $(hostname) at $(date)"

# cd into the project folder
cd /home/marcel42/kidney-tumor-radiomics

# 1. Point to your custom installed libraries
export PYTHONUSERBASE=/home/marcel42/kidney-tumor-radiomics/env_libs
export PATH=$PYTHONUSERBASE/bin:$PATH

# 2. INSTALL the libraries (this automatically creates the env_libs folder)
apptainer exec --nv /lu/tetyda/projects/center4ml/Containers/tensorflow-gpu.sif pip3 install --user -r requirements.txt

# 3. Run python script within container (using the newly installed libraries)
apptainer exec --nv /lu/tetyda/projects/center4ml/Containers/tensorflow-gpu.sif python3 train.py --data_dir /home/marcel42/Data/kits23_processed/kits_dataset_16_30_30_kidney --out_dir /home/marcel42/kidney-tumor-radiomics/HPC_out --epochs 320 --n_folds 5 --workers 8 --save_model --architecture anisotropic_reg_0  --use_lr_scheduler

# Print completion status
echo "Job finished at $(date)"