#!/bin/bash
####### Reserve computing resources #############
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=23:00:00
#SBATCH --mem=0
#SBATCH --partition=gpu-a100 --gres=gpu:2
#SBATCH --job-name=UNET3D
#SBATCH --mail-user=njneetes@ucalgary.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

####### Set environment variables ###############
export PATH="$HOME/software/miniconda3/bin:$PATH"

####### Run your script #########################
source activate blptl_11.1
python -u python/training/final/train_unet_final.py \
unet_3d_final version_21096524 \
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
--label unet_3d_knee_wback_transfer_final \
--version "$SLURM_JOB_ID" \
--batch-size 16 \
--log-step-interval 3 \
--cuda \
--num-workers 4 \
--hfn "ref_hparams.yaml"