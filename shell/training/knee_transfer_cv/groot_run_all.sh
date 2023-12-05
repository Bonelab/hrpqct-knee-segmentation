#!/bin/bash
python -u python/training/knee_cv/train_unet_knee_cv.py \
unet_3d_final version_21096524 \
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
/home/njneetes/scratch/auto_peri_knee/data/normxtii/preprocessed/tibia_pickled_3d_background \
--label unet_3d_knee_wback_e100_transfer_cv \
--version "$SLURM_JOB_ID" \
--folds 5 \
--num-workers 6 \
--batch-size 16 \
--log-step-interval 3 \
--cuda &> unet_transfer.log
python -u python/training/knee_cv/train_unet_knee_cv.py \
unetpp_3d_final version_21096525 \
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
/home/njneetes/scratch/auto_peri_knee/data/normxtii/preprocessed/tibia_pickled_3d_background \
--label unetpp_3d_knee_wback_e100_transfer_cv \
--version "$SLURM_JOB_ID" \
--folds 5 \
--num-workers 6 \
--batch-size 16 \
--log-step-interval 3 \
--cuda &> unetpp_transfer.log
python -u python/training/knee_cv/train_unet_knee_cv.py \
unetr_3d_final version_21096526 \
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
/home/njneetes/scratch/auto_peri_knee/data/normxtii/preprocessed/tibia_pickled_3d_background \
--label unetr_3d_knee_wback_e100_transfer_cv \
--version "$SLURM_JOB_ID" \
--folds 5 \
--num-workers 6 \
--batch-size 16 \
--log-step-interval 3 \
--cuda &> unetr_transfer.log
python -u python/training/knee_cv/train_segan_knee_cv.py \
segan_3d_final version_21096520 \
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
/home/njneetes/scratch/auto_peri_knee/data/normxtii/preprocessed/tibia_pickled_3d_background \
--label segan_3d_knee_wback_e100_transfer_cv \
--version "$SLURM_JOB_ID" \
--folds 5 \
--num-workers 6 \
--batch-size 16 \
--log-step-interval 3 \
--cuda &> segan_transfer.log
python -u python/training/knee_cv/train_segresnetvae_knee_cv.py \
segresnetvae_3d_final version_21096522 \
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
/home/njneetes/scratch/auto_peri_knee/data/normxtii/preprocessed/tibia_pickled_3d_background \
--label segresnetvae_3d_knee_wback_e100_transfer_cv \
--version "$SLURM_JOB_ID" \
--folds 5 \
--num-workers 6 \
--batch-size 128 \
--log-step-interval 3 \
--cuda &> segresnetvae_transfer.log