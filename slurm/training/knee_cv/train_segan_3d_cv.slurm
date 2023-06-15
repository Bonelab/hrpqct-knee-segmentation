#!/bin/bash
####### Reserve computing resources #############
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --time=23:00:00
#SBATCH --mem=0
#SBATCH --partition=gpu-a100 --gres=gpu:2
#SBATCH --job-name=SEGAN3D
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
python -u python/training/cv/train_segan_cv.py \
--label segan_3d_knee_base_cv \
--version "$SLURM_JOB_ID" \
--input-channels 1 \
--output-channels 3 \
--model-channels ${CHANNELS_ARR[@]} \
--batch-size 16 \
--dropout "$DROPOUT" \
--learning-rate "$LEARNING_RATE" \
--log-step-interval 3 \
--cuda \
--folds 5 \
--num-workers 6 \
--upsample-mode trilinear --is-3d \
/home/njneetes/work/data/SALTAC/visit_1/femur/preprocessed/medial_pickled_3d \
/home/njneetes/work/data/SALTAC/visit_1/femur/preprocessed/lateral_pickled_3d \
/home/njneetes/work/data/SALTAC/visit_1/tibia/preprocessed/medial_pickled_3d \
/home/njneetes/work/data/SALTAC/visit_1/tibia/preprocessed/lateral_pickled_3d \
/home/njneetes/work/data/SALTAC/visit_2/femur/preprocessed/medial_pickled_3d \
/home/njneetes/work/data/SALTAC/visit_2/femur/preprocessed/lateral_pickled_3d \
/home/njneetes/work/data/SALTAC/visit_2/tibia/preprocessed/medial_pickled_3d \
/home/njneetes/work/data/SALTAC/visit_2/tibia/preprocessed/lateral_pickled_3d \
/home/njneetes/work/data/SALTAC/visit_3/femur/preprocessed/medial_pickled_3d \
/home/njneetes/work/data/SALTAC/visit_3/femur/preprocessed/lateral_pickled_3d \
/home/njneetes/work/data/SALTAC/visit_3/tibia/preprocessed/medial_pickled_3d \
/home/njneetes/work/data/SALTAC/visit_3/tibia/preprocessed/lateral_pickled_3d \
/home/njneetes/work/data/SALTAC/visit_4/femur/preprocessed/medial_pickled_3d \
/home/njneetes/work/data/SALTAC/visit_4/femur/preprocessed/lateral_pickled_3d \
/home/njneetes/work/data/SALTAC/visit_4/tibia/preprocessed/medial_pickled_3d \
/home/njneetes/work/data/SALTAC/visit_4/tibia/preprocessed/lateral_pickled_3d
