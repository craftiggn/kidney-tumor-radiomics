#!/bin/bash -l
# run with: sbatch compile_dataset.sh

# --- SLURM DIRECTIVES (Resource Requests) ---

#SBATCH --job-name=compile_dataset
#SBATCH --account=g105-2708
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=dataset_job_%j.log

# --- SCRIPT COMMANDS ---

# print start information on the log file
echo "Job started on $(hostname) at $(date)"

# cd into the project folder
cd /home/marcel42/kidney-tumor-radiomics

# run python script within container
apptainer exec --nv /lu/tetyda/projects/center4ml/Containers/tensorflow-gpu.sif python3 compile_dataset.py

# 4. Print completion status
echo "Job finished at $(date)"