#!/bin/bash
####### Reserve computing resources #############
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=6:00:00
#SBATCH --mem=150G
#SBATCH --job-name=4_LONG_REG
#SBATCH --mail-user=njneetes@ucalgary.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

####### Set environment variables ###############
export PATH="$HOME/software/miniconda3/bin:$PATH"
export PYTHONUNBUFFERED=1
export NIFTI_DIR="/home/njneetes/work/data/TRIKNEE/niftis"
export MODEL_MASK_DIR="/home/njneetes/work/data/TRIKNEE/model_masks"
export REG_DIR="/home/njneetes/work/data/TRIKNEE/registrations"

####### Run your script #########################
source activate blptl_11.1
echo "Step 1: mask the baseline image with the postprocessed model mask"
python python/preprocessing/mask_image.py \
${NIFTI_DIR}/${LABEL,,}_1.nii.gz \
${MODEL_MASK_DIR}/${LABEL,,}_1_postprocessed_mask.nii.gz \
${NIFTI_DIR}/${LABEL,,}_1_masked.nii.gz \
--dilate-amount 35 --background-class 0 --background-value -1000 -ow
echo "Step 2: mask the followup images with the postprocessed model mask"
python python/preprocessing/mask_image.py \
${NIFTI_DIR}/${LABEL,,}_2.nii.gz \
${MODEL_MASK_DIR}/${LABEL,,}_2_postprocessed_mask.nii.gz \
${NIFTI_DIR}/${LABEL,,}_2_masked.nii.gz \
--dilate-amount 35 --background-class 0 --background-value -1000 -ow
python python/preprocessing/mask_image.py \
${NIFTI_DIR}/${LABEL,,}_3.nii.gz \
${MODEL_MASK_DIR}/${LABEL,,}_3_postprocessed_mask.nii.gz \
${NIFTI_DIR}/${LABEL,,}_3_masked.nii.gz \
--dilate-amount 35 --background-class 0 --background-value -1000 -ow
echo "Step 3: Register the followup image to the baseline image"
blRegistrationLongitudinal \
${REG_DIR} ${LABEL,,} \
${NIFTI_DIR}/${LABEL,,}_1_masked.nii.gz \
${NIFTI_DIR}/${LABEL,,}_2_masked.nii.gz \
${NIFTI_DIR}/${LABEL,,}_3_masked.nii.gz \
--baseline-label 1 --follow-up-labels 2 3 \
--max-iterations 1000 \
--shrink-factors 16 8 4 2 1 --smoothing-sigmas 8 4 2 1 0.1 \
--plot-metric-history \
--optimizer GradientDescent --gradient-descent-learning-rate 0.0005 \
--gradient-descent-convergence-min-value 0.000001 \
--gradient-descent-convergence-window-size 10 \
--similarity-metric Correlation \
--similarity-metric-sampling-strategy Random --similarity-metric-sampling-rate 0.01 \
--interpolator Linear \
--centering-initialization Geometry \
--overwrite
echo "Step 4: Remove the masked images"
rm ${NIFTI_DIR}/${LABEL,,}_1_masked.nii.gz
rm ${NIFTI_DIR}/${LABEL,,}_2_masked.nii.gz
rm ${NIFTI_DIR}/${LABEL,,}_3_masked.nii.gz
