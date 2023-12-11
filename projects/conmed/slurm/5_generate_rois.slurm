#!/bin/bash
####### Reserve computing resources #############
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=2:00:00
#SBATCH --mem=150G
#SBATCH --job-name=5_GEN_ROIS
#SBATCH --mail-user=njneetes@ucalgary.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

####### Set environment variables ###############
export PATH="$HOME/software/miniconda3/bin:$PATH"
export PYTHONUNBUFFERED=1
export ATLAS_REG_DIR="/home/njneetes/work/data/CONMED/atlas_registrations"
export ROI_MASK_DIR="/home/njneetes/work/data/CONMED/roi_masks"
export MODEL_MASK_DIR="/home/njneetes/work/data/CONMED/model_masks"

####### Run your script #########################
source activate blptl_11.1
python python/generate_rois/generate_rois.py \
${MODEL_MASK_DIR}/${IMAGE,,}_postprocessed_mask.nii.gz ${BONE} \
${ATLAS_REG_DIR}/${IMAGE,,}_atlas_mask_transformed.nii.gz \
${ROI_MASK_DIR} ${IMAGE,,} -ow