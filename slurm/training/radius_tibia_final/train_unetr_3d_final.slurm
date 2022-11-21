#!/bin/bash
####### Reserve computing resources #############
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=23:00:00
#SBATCH --mem=0
#SBATCH --partition=gpu-a100 --gres=gpu:2
#SBATCH --job-name=UNETR3D
#SBATCH --mail-user=njneetes@ucalgary.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

####### Set environment variables ###############
export PATH="$HOME/software/miniconda3/bin:$PATH"

####### Run your script #########################
source activate blptl_11.1
python -u python/training/final/train_unet_final.py \
unetr_3d_cv 16639306_f0 \
/home/njneetes/work/data/NORMXTII/radius_pickled_3d \
/home/njneetes/work/data/NORMXTII/tibia_pickled_3d \
/home/njneetes/work/data/HIPFX/radius_pickled_3d \
/home/njneetes/work/data/HIPFX/tibia_pickled_3d \
--label unetr_3d_final \
--version "$SLURM_JOB_ID" \
--batch-size 16 \
--log-step-interval 3 \
--cuda \
--num-workers 4