#!/bin/bash
####### Reserve computing resources #############
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=23:00:00
#SBATCH --mem=0
#SBATCH --partition=gpu-a100 --gres=gpu:2
#SBATCH --job-name=SRNVAE3D
#SBATCH --mail-user=njneetes@ucalgary.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

####### Set environment variables ###############
export PATH="$HOME/software/miniconda3/bin:$PATH"

####### Run your script #########################
source activate blptl_11.1
python -u python/training/knee_cv/train_segresnetvae_knee_cv.py \
segresnetvae_3d_final version_21096522 \
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
--label segresnetvae_3d_knee_transfer_cv \
--version "$SLURM_JOB_ID" \
--folds 5 \
--num-workers 6 \
--batch-size 128 \
--log-step-interval 3 \
--cuda