#!/bin/bash
####### Reserve computing resources #############
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=23:00:00
#SBATCH --mem=200G
#SBATCH --partition=gpu-a100 --gres=gpu:2
#SBATCH --job-name=UNETPP2D
#SBATCH --mail-user=njneetes@ucalgary.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

####### Set environment variables ###############
export PATH="$HOME/software/miniconda3/bin:$PATH"

####### Run your script #########################
source activate blptl_11.1
CHANNELS_ARR=("$CHANNELS")
# shellcheck disable=SC2068
python -u python/training/cv/train_unet_cv.py \
--model-architecture unet++ \
--label unetpp_2d_cv \
--version "$SLURM_JOB_ID" \
--input-channels 1 \
--output-channels 3 \
--model-channels ${CHANNELS_ARR[@]} \
--batch-size 256 \
--dropout "$DROPOUT" \
--learning-rate "$LEARNING_RATE" \
--log-step-interval 3 \
--cuda \
--folds 5 \
--num-workers 4 \
/home/njneetes/work/data/NORMXTII/radius_pickled_2d \
/home/njneetes/work/data/NORMXTII/tibia_pickled_2d \
/home/njneetes/work/data/HIPFX/radius_pickled_2d \
/home/njneetes/work/data/HIPFX/tibia_pickled_2d
