#!/bin/bash
####### Reserve computing resources #############
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --time=23:00:00
#SBATCH --mem=200G
#SBATCH --partition=gpu-a100 --gres=gpu:2
#SBATCH --job-name=UNET_GS
#SBATCH --mail-user=njneetes@ucalgary.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

####### Set environment variables ###############
export PATH="$HOME/software/miniconda3/bin:$PATH"

####### Run your script #########################
rm -rf /dev/shm/* &>/dev/null
source activate blptl_11.1
CHANNELS_ARR=("$CHANNELS")
# shellcheck disable=SC2068
python -u python/training/train_unet_cv.py \
--label unet_3d_cv \
--version "$SLURM_JOB_ID" \
--input-channels 1 \
--output-channels 3 \
--model-channels ${CHANNELS_ARR[@]} \
--dropout "$DROPOUT" \
--epochs 250 \
--log-step-interval 1 \
--num-gpus 2 \
--folds 5 \
--num-workers 5 \
--auto-learning-rate \
--is-3d \
/home/njneetes/work/data/NORMXTII/radius_pickled_2d \
/home/njneetes/work/data/NORMXTII/tibia_pickled_2d \
/home/njneetes/work/data/HIPFX/radius_pickled_2d \
/home/njneetes/work/data/HIPFX/tibia_pickled_2d
rm -rf /dev/shm/* &>/dev/null