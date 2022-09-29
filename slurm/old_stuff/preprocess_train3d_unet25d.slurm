#!/bin/bash
####### Reserve computing resources #############
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --job-name=Preprocess_3D
#SBATCH --mail-user=njneetes@ucalgary.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

####### Set environment variables ###############
export PATH="$HOME/software/miniconda3/bin:$PATH"

####### Run your script #########################
source activate blptl
python -u python/preprocess_3d_patch_samples_w_2d_preds.py /home/njneetes/Data/HRpQCT/Train /home/njneetes/Data/HRpQCT/Train_pickled_3D -e 1
python -u python/preprocess_3d_patch_samples_w_2d_preds.py /home/njneetes/Data/HRpQCT/Validate /home/njneetes/Data/HRpQCT/Validate_pickled_3D -e 1
