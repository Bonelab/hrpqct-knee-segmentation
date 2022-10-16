#!/bin/bash
####### Reserve computing resources #############
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --time=12:00:00
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
python -u python/training/train_unet_2d_cv.py \
--label unet_2d_cv \
--version "$SLURM_JOB_ID" \
--channels "$CHANNELS" \
--dropout "$DROPOUT" \
--learning-rate "$LEARNING_RATE" \
--epochs 250 \
--batch-size 32 \
--num-gpus 1 \
--folds 5 \
--workers 5 \
/home/njneetes/work/data/NORMXTII/radius_pickled_2d \
/home/njneetes/work/data/NORMXTII/tibia_pickled_2d \
/home/njneetes/work/data/HIPFX/radius_pickled_2d \
/home/njneetes/work/data/HIPFX/tibia_pickled_2d

