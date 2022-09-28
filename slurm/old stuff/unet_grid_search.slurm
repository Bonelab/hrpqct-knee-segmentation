#!/bin/bash
####### Reserve computing resources #############
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --time=23:00:00
#SBATCH --mem=300G
#SBATCH --partition=gpu-v100 --gres=gpu:1
#SBATCH --job-name=UNET_GS
#SBATCH --mail-user=njneetes@ucalgary.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

####### Set environment variables ###############
export PATH="$HOME/software/miniconda3/bin:$PATH"

####### Run your script #########################
source activate blptl
python -u python/unet_2d_grid_search.py \
-td ~/Data/HRpQCT/Train_pickled \
-vd ~/Data/HRpQCT/Validate_pickled \
-ld logs \
-bs 64 -e 100 -w 10 -ng 1 -l unet2d_multi_gs
