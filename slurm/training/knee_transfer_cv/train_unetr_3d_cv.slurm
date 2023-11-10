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
python -u python/training/knee_cv/train_unet_knee_cv.py \
unetr_3d_final version_21096526 \
/home/njneetes/work/data/SALTAC/visit_1/preprocessed/femur_medial_pickled_3d \
/home/njneetes/work/data/SALTAC/visit_1/preprocessed/femur_lateral_pickled_3d \
/home/njneetes/work/data/SALTAC/visit_1/preprocessed/tibia_medial_pickled_3d \
/home/njneetes/work/data/SALTAC/visit_1/preprocessed/tibia_lateral_pickled_3d \
/home/njneetes/work/data/SALTAC/visit_2/preprocessed/femur_medial_pickled_3d \
/home/njneetes/work/data/SALTAC/visit_2/preprocessed/femur_lateral_pickled_3d \
/home/njneetes/work/data/SALTAC/visit_2/preprocessed/tibia_medial_pickled_3d \
/home/njneetes/work/data/SALTAC/visit_2/preprocessed/tibia_lateral_pickled_3d \
/home/njneetes/work/data/SALTAC/visit_3/preprocessed/femur_medial_pickled_3d \
/home/njneetes/work/data/SALTAC/visit_3/preprocessed/femur_lateral_pickled_3d \
/home/njneetes/work/data/SALTAC/visit_3/preprocessed/tibia_medial_pickled_3d \
/home/njneetes/work/data/SALTAC/visit_3/preprocessed/tibia_lateral_pickled_3d \
/home/njneetes/work/data/SALTAC/visit_4/preprocessed/femur_medial_pickled_3d \
/home/njneetes/work/data/SALTAC/visit_4/preprocessed/femur_lateral_pickled_3d \
/home/njneetes/work/data/SALTAC/visit_4/preprocessed/tibia_medial_pickled_3d \
/home/njneetes/work/data/SALTAC/visit_4/preprocessed/tibia_lateral_pickled_3d \
/home/njneetes/work/data/NORMXTII/preprocessed/radius_pickled_3d_background \
/home/njneetes/work/data/NORMXTII/preprocessed/tibia_pickled_3d_background \
--label unetr_3d_knee_wback_transfer_cv \
--version "$SLURM_JOB_ID" \
--folds 5 \
--num-workers 6 \
--batch-size 16 \
--log-step-interval 3 \
--cuda