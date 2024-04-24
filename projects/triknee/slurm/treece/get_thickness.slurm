#!/bin/bash
####### Reserve computing resources #############
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=6:00:00
#SBATCH --mem=64G
#SBATCH --job-name=TREECE
#SBATCH --mail-user=njneetes@ucalgary.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

####### Set environment variables ###############
export PATH="$HOME/software/miniconda3/bin:$PATH"
export PYTHONUNBUFFERED=1
export BASE_DIR="/home/njneetes/work/data/TRIKNEE"
export NIFTI_DIR="niftis"
export TREECE_DIR="treece"
export MODEL_MASK_DIR="model_masks"
export ROI_DIR_BASE="roi_masks"

####### Run your script #########################
source activate bl
blTreeceThickness \
${BASE_DIR}/${NIFTI_DIR}/${IMAGE,,}.nii.gz \
${BASE_DIR}/${TREECE_DIR}/${IMAGE,,}_${COMPARTMENT}_${REG} \
-bm ${BASE_DIR}/${MODEL_MASK_DIR}/${IMAGE,,}_postprocessed_mask.nii.gz \
-sm ${BASE_DIR}/${ROI_DIR_BASE}_${REG}/${IMAGE,,}_allrois_mask.nii.gz \
-sml ${ROI_CODE} \
-ss -ssi 30  -iss 2.0 \
-lr 0.06 -rbf 3.0 \
-sod 3 -sid 9 -stig 0 -tbig 0 -ow -mst \
-tig 0.7 -tb 0.244 10.0
