#!/bin/bash -l
# run with: sbatch build_container_image.sh
# remember to make sure that the container image (.sif) is not tracked by git as it is too heavy

# --- SLURM DIRECTIVES (Resource Requests) ---

#SBATCH --job-name=build_container
#SBATCH --account=g105-2708
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=container_job_%j.log

# --- SCRIPT COMMANDS ---

# print start information on the log file
echo "Job started on $(hostname) at $(date)"

# cd into the project folder
cd /home/marcel42/kidney-tumor-radiomics

# run python script within container
module load apptainer
apptainer build --fakeroot container.sif container.def

# 4. Print completion status
echo "Container build successfully"