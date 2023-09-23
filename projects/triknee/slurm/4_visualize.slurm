#!/bin/bash
####### Reserve computing resources #############
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=0:30:00
#SBATCH --mem=100G
#SBATCH --job-name=Post-inference
#SBATCH --mail-user=njneetes@ucalgary.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

####### Set environment variables ###############
export PATH="$HOME/software/miniconda3/bin:$PATH"
export PYTHONUNBUFFERED=1
export NIFTI_DIR="/home/njneetes/work/data/TRIKNEE/niftis"
export ROI_MASK_DIR="/home/njneetes/work/data/TRIKNEE/roi_masks"
export VIZ_DIR="/home/njneetes/work/data/TRIKNEE/visualizations"

####### Run your script #########################
source activate blptl_11.1
python python/visualization/write_panning_video.py \
${NIFTI_DIR}/${IMAGE,,}.nii.gz \
${ROI_MASK_DIR}/${IMAGE,,}_allrois_mask.nii.gz \
${VIZ_DIR}/${IMAGE,,} \
-ib -400 1400 -pd 1 -ri -cens 10