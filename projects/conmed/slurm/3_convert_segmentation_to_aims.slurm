#!/bin/bash
####### Reserve computing resources #############
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=2:00:00
#SBATCH --mem=150G
#SBATCH --job-name=3_CONVERT_TO_AIM
#SBATCH --mail-user=njneetes@ucalgary.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

####### Set environment variables ###############
export PATH="$HOME/software/miniconda3/bin:$PATH"
export PYTHONUNBUFFERED=1
export AIM_DIR="/home/njneetes/work/data/CONMED/aims"
export MODEL_MASK_DIR="/home/njneetes/work/data/CONMED/model_masks"
export ROI_MASK_DIR="/home/njneetes/work/data/CONMED/roi_masks"

####### Run your script #########################
source activate blptl_11.1
echo "Convert cort / trab/ tunnel mask nifti to AIMs"
python python/aim_nifti/convert_masks_to_aims.py \
${MODEL_MASK_DIR}/${IMAGE,,}_postprocessed_mask.nii.gz \
${AIM_DIR}/${IMAGE}.AIM \
${ROI_MASK_DIR}/${IMAGE} \
-cv 1 2 3 \
-cl CORT_MASK TRAB_MASK TUNNEL_MASK \
-l "Generated using the code at: https://github.com/Bonelab/hrpqct-knee-segmentation" -ow