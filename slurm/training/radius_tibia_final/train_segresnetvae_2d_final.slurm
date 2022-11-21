#!/bin/bash
####### Reserve computing resources #############
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=23:00:00
#SBATCH --mem=300G
#SBATCH --partition=gpu-a100 --gres=gpu:2
#SBATCH --job-name=SRNVAE2D
#SBATCH --mail-user=njneetes@ucalgary.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

####### Set environment variables ###############
export PATH="$HOME/software/miniconda3/bin:$PATH"

####### Run your script #########################
source activate blptl_11.1
python -u python/training/final/train_segresnetvae_final.py \
segresnetvae_2d_cv 16636140_f0 \
/home/njneetes/work/data/NORMXTII/radius_pickled_2d \
/home/njneetes/work/data/NORMXTII/tibia_pickled_2d \
/home/njneetes/work/data/HIPFX/radius_pickled_2d \
/home/njneetes/work/data/HIPFX/tibia_pickled_2d \
--label segresnetvae_2d_final \
--version "$SLURM_JOB_ID" \
--batch-size 128 \
--log-step-interval 3 \
--cuda \
--num-workers 4