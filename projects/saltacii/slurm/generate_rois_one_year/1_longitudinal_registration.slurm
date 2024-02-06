#!/bin/bash
####### Reserve computing resources #############
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=6:00:00
#SBATCH --mem=150G
#SBATCH --job-name=1_LONG_REG
#SBATCH --mail-user=njneetes@ucalgary.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

####### Set environment variables ###############
export PATH="$HOME/software/miniconda3/bin:$PATH"
export PYTHONUNBUFFERED=1
export NIFTI_DIR="/home/njneetes/work/data/SALTACII/niftis"
export MODEL_MASK_DIR="/home/njneetes/work/data/SALTACII/model_masks"
export REG_DIR="/home/njneetes/work/data/SALTACII/registrations"

####### Run your script #########################
source activate blptl_11.1
echo "Step 1: mask the baseline image with the postprocessed model mask"
python python/preprocessing/mask_image.py \
${NIFTI_DIR}/${LABEL,,}_${T0,,}.nii.gz \
${MODEL_MASK_DIR}/${LABEL,,}_${T0,,}_postprocessed_mask.nii.gz \
${NIFTI_DIR}/${LABEL,,}_${T0,,}_masked.nii.gz \
--dilate-amount 35 --background-class 0 --background-value -1000 -ow
echo "Step 2: mask the followup images with the postprocessed model mask"
python python/preprocessing/mask_image.py \
${NIFTI_DIR}/${LABEL,,}_${T1,,}.nii.gz \
${MODEL_MASK_DIR}/${LABEL,,}_${T1,,}_postprocessed_mask.nii.gz \
${NIFTI_DIR}/${LABEL,,}_${T1,,}_masked.nii.gz \
--dilate-amount 35 --background-class 0 --background-value -1000 -ow
echo "Step 3: Register the followup image to the baseline image"
blRegistrationLongitudinal \
${REG_DIR} ${LABEL,,} \
${NIFTI_DIR}/${LABEL,,}_${T0,,}_masked.nii.gz \
${NIFTI_DIR}/${LABEL,,}_${T1,,}_masked.nii.gz \
--baseline-label ${T0,,} --follow-up-labels ${T1,,} \
--max-iterations 1000 \
--downsampling-shrink-factor 4 \
--downsampling-smoothing-sigma 1 \
--shrink-factors 16 8 4 2 1 --smoothing-sigmas 8 4 2 1 0.1 \
--plot-metric-history \
--optimizer Powell \
--similarity-metric Correlation \
--similarity-metric-sampling-strategy Regular --similarity-metric-sampling-rate 0.3 \
--interpolator Linear \
--centering-initialization Geometry \
--overwrite
echo "Step 4: Remove the masked images"
rm ${NIFTI_DIR}/${LABEL,,}_${T0,,}_masked.nii.gz
rm ${NIFTI_DIR}/${LABEL,,}_${T1,,}_masked.nii.gz
