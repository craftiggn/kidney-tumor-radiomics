# run this batch file to build a container image, on an HPC (ICM)
apptainer build --fakeroot container.sif container.def
# remember to make sure that the container image (.sif) is not tracked by git as it is too heavy