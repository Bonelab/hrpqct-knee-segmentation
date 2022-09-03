#!/bin/bash
####### Reserve computing resources #############
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --time=12:00:00
#SBATCH --mem=300G
#SBATCH --partition=gpu-v100 --gres=gpu:1
#SBATCH --job-name=Synthesis_GS_Test
#SBATCH --mail-user=njneetes@ucalgary.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

####### Set environment variables ###############
export PATH="$HOME/software/miniconda3/bin:$PATH"

####### Run your script #########################
source activate blptl
python -u python/synthesis_3d_grid_search.py \
-td ~/Data/HRpQCT/Train_pickled_3D \
-vd ~/Data/HRpQCT/Validate_pickled_3D \
-l synthesis3d_gs \
-bs 4 -e 100 -ng 1 -w 10 \
-v ${SLURM_JOB_ID} \
-mf FILTERS \
-c CHANNELS \
-d DROPOUT
