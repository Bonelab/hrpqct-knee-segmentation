#!/bin/bash
####### Reserve computing resources #############
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=2:00:00
#SBATCH --mem=100G
#SBATCH --job-name=Convert-to-AIM
#SBATCH --mail-user=njneetes@ucalgary.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

####### Set environment variables ###############
export PATH="$HOME/software/miniconda3/bin:$PATH"
export PYTHONUNBUFFERED=1
export AIM_DIR="/home/njneetes/work/data/TRIKNEE/aims"
export ROI_MASK_DIR="/home/njneetes/work/data/TRIKNEE/roi_masks"

####### Run your script #########################
source activate blptl_11.1
echo "Convert roi mask nifti to AIM"
python python/aim_nifti/convert_mask_to_aim.py \
${ROI_MASK_DIR}/${IMAGE,,}_roi${ROI_CODE}_mask.nii.gz \
${AIM_DIR}/${IMAGE}.AIM \
${ROI_MASK_DIR}/${IMAGE}_ROI${ROI_CODE}_MASK.AIM \
-l "Generated using the code at: https://github.com/Bonelab/hrpqct-knee-segmentation" -ow