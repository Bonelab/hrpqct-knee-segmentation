#!/bin/bash
####### Reserve computing resources #############
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --job-name=Preprocess_Train
#SBATCH --mail-user=njneetes@ucalgary.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

####### Set environment variables ###############
export PATH="$HOME/software/miniconda3/bin:$PATH"

####### Run your script #########################
source activate blptl_11.1
python -u python/inference/inference_unet.py \
/home/njneetes/Projects/hrpqct-knee-segmentation/logs/unetr_3d_final/version_16826609/ref_hparams.yaml \
/home/njneetes/Projects/hrpqct-knee-segmentation/logs/unetr_3d_knee_cv/17172956_f4/checkpoints/epoch=512-step=6669.ckpt \
"$DATA_DIR"/"$IMAGE".AIM \
"$DATA_DIR"/inference/"$IMAGE"_SCBP_MASK.AIM \
"$DATA_DIR"/inference/"$IMAGE"_TRAB_MASK.AIM \
