#!/bin/bash
CHANNELS=$1
DROPOUT=$2
LEARNING_RATE=$3
CHANNELS_ARR=("$CHANNELS")
# shellcheck disable=SC2068
python -u python/training/cv/train_unet_cv.py \
--label unet_3d_knee_wback_e100_base_cv \
--input-channels 1 \
--output-channels 3 \
--model-channels ${CHANNELS_ARR[@]} \
--batch-size 16 \
--dropout "$DROPOUT" \
--learning-rate "$LEARNING_RATE" \
--log-step-interval 3 \
--cuda \
--folds 5 \
--num-workers 4 \
--is-3d \
/home/njneetes/scratch/auto_peri_knee/data/saltac/visit_1/preprocessed/femur_medial_pickled_3d_e100 \
/home/njneetes/scratch/auto_peri_knee/data/saltac/visit_1/preprocessed/femur_lateral_pickled_3d_e100 \
/home/njneetes/scratch/auto_peri_knee/data/saltac/visit_1/preprocessed/tibia_medial_pickled_3d_e100 \
/home/njneetes/scratch/auto_peri_knee/data/saltac/visit_1/preprocessed/tibia_lateral_pickled_3d_e100 \
/home/njneetes/scratch/auto_peri_knee/data/saltac/visit_2/preprocessed/femur_medial_pickled_3d_e100 \
/home/njneetes/scratch/auto_peri_knee/data/saltac/visit_2/preprocessed/femur_lateral_pickled_3d_e100 \
/home/njneetes/scratch/auto_peri_knee/data/saltac/visit_2/preprocessed/tibia_medial_pickled_3d_e100 \
/home/njneetes/scratch/auto_peri_knee/data/saltac/visit_2/preprocessed/tibia_lateral_pickled_3d_e100 \
/home/njneetes/scratch/auto_peri_knee/data/saltac/visit_3/preprocessed/femur_medial_pickled_3d_e100 \
/home/njneetes/scratch/auto_peri_knee/data/saltac/visit_3/preprocessed/femur_lateral_pickled_3d_e100 \
/home/njneetes/scratch/auto_peri_knee/data/saltac/visit_3/preprocessed/tibia_medial_pickled_3d_e100 \
/home/njneetes/scratch/auto_peri_knee/data/saltac/visit_3/preprocessed/tibia_lateral_pickled_3d_e100 \
/home/njneetes/scratch/auto_peri_knee/data/saltac/visit_4/preprocessed/femur_medial_pickled_3d_e100 \
/home/njneetes/scratch/auto_peri_knee/data/saltac/visit_4/preprocessed/femur_lateral_pickled_3d_e100 \
/home/njneetes/scratch/auto_peri_knee/data/saltac/visit_4/preprocessed/tibia_medial_pickled_3d_e100 \
/home/njneetes/scratch/auto_peri_knee/data/saltac/visit_4/preprocessed/tibia_lateral_pickled_3d_e100 \
/home/njneetes/scratch/auto_peri_knee/data/normxtii/preprocessed/radius_pickled_3d_background \
/home/njneetes/scratch/auto_peri_knee/data/normxtii/preprocessed/tibia_pickled_3d_background