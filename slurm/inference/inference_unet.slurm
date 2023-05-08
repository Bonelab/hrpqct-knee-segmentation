#!/bin/bash
####### Reserve computing resources #############
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=6:00:00
#SBATCH --mem=64G
#SBATCH --job-name=Inference_Unet
#SBATCH --mail-user=njneetes@ucalgary.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

####### Set environment variables ###############
export PATH="$HOME/software/miniconda3/bin:$PATH"

####### Run your script #########################
source activate blptl_11.1
python -u python/inference/inference_unet.py \
/home/njneetes/Projects/hrpqct-knee-segmentation/logs/unetpp_3d_final/version_16825523/ref_hparams.yaml \
/home/njneetes/Projects/hrpqct-knee-segmentation/logs/unetpp_3d_knee_cv/17172953_f4/checkpoints/epoch=144-step=1885.ckpt \
"$DATA_DIR"/"$IMAGE".AIM \
"$DATA_DIR"/inference_unetpp/"$IMAGE"_SCBP_MASK.AIM \
"$DATA_DIR"/inference_unetpp/"$IMAGE"_TRAB_MASK.AIM \
-pw 128
