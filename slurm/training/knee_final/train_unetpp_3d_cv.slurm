#!/bin/bash
####### Reserve computing resources #############
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=23:00:00
#SBATCH --mem=0
#SBATCH --partition=gpu-a100 --gres=gpu:2
#SBATCH --job-name=UNETPP3D
#SBATCH --mail-user=njneetes@ucalgary.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

####### Set environment variables ###############
export PATH="$HOME/software/miniconda3/bin:$PATH"

####### Run your script #########################
source activate blptl_11.1
python -u python/training/knee_cv/train_unet_knee_cv.py \
unetpp_3d_final version_16825523 \
/home/njneetes/work/data/SALTAC/visit_1/femur/medial_pickled_3d \
/home/njneetes/work/data/SALTAC/visit_1/femur/lateral_pickled_3d \
/home/njneetes/work/data/SALTAC/visit_1/tibia/medial_pickled_3d \
/home/njneetes/work/data/SALTAC/visit_1/tibia/lateral_pickled_3d \
/home/njneetes/work/data/SALTAC/visit_2/femur/medial_pickled_3d \
/home/njneetes/work/data/SALTAC/visit_2/femur/lateral_pickled_3d \
/home/njneetes/work/data/SALTAC/visit_2/tibia/medial_pickled_3d \
/home/njneetes/work/data/SALTAC/visit_2/tibia/lateral_pickled_3d \
/home/njneetes/work/data/SALTAC/visit_3/femur/medial_pickled_3d \
/home/njneetes/work/data/SALTAC/visit_3/femur/lateral_pickled_3d \
/home/njneetes/work/data/SALTAC/visit_3/tibia/medial_pickled_3d \
/home/njneetes/work/data/SALTAC/visit_3/tibia/lateral_pickled_3d \
/home/njneetes/work/data/SALTAC/visit_4/femur/medial_pickled_3d \
/home/njneetes/work/data/SALTAC/visit_4/femur/lateral_pickled_3d \
/home/njneetes/work/data/SALTAC/visit_4/tibia/medial_pickled_3d \
/home/njneetes/work/data/SALTAC/visit_4/tibia/lateral_pickled_3d \
--label unetpp_3d_knee_cv \
--version "$SLURM_JOB_ID" \
--folds 5 \
--num-workers 6 \
--batch-size 16 \
--log-step-interval 3