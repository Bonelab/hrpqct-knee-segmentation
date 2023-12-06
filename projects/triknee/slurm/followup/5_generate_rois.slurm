#!/bin/bash
####### Reserve computing resources #############
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=5:00:00
#SBATCH --mem=150G
#SBATCH --job-name=5_GEN_ROIS
#SBATCH --mail-user=njneetes@ucalgary.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

####### Set environment variables ###############
export PATH="$HOME/software/miniconda3/bin:$PATH"
export PYTHONUNBUFFERED=1
export ATLAS_REG_DIR="/home/njneetes/work/data/TRIKNEE/atlas_registrations"
export REG_DIR="/home/njneetes/work/data/TRIKNEE/registrations"
export ROI_MASK_DIR="/home/njneetes/work/data/TRIKNEE/roi_masks_reg"
export MODEL_MASK_DIR="/home/njneetes/work/data/TRIKNEE/model_masks"

####### Run your script #########################
source activate blptl_11.1
echo "Step 1: Transform the followup image segmentations to the baseline reference frame"
blRegistrationApplyTransform \
${MODEL_MASK_DIR}/${LABEL,,}_2_postprocessed_mask.nii.gz \
${REG_DIR}/${LABEL,,}_2_transform.txt \
${MODEL_MASK_DIR}/${LABEL,,}_2_transformed_mask.nii.gz \
-fi ${MODEL_MASK_DIR}/${LABEL,,}_1_postprocessed_mask.nii.gz \
-int NearestNeighbour -ow
blRegistrationApplyTransform \
${MODEL_MASK_DIR}/${LABEL,,}_3_postprocessed_mask.nii.gz \
${REG_DIR}/${LABEL,,}_3_transform.txt \
${MODEL_MASK_DIR}/${LABEL,,}_3_transformed_mask.nii.gz \
-fi ${MODEL_MASK_DIR}/${LABEL,,}_1_postprocessed_mask.nii.gz \
-int NearestNeighbour -ow
echo "Step 2: Generate the ROIs in the baseline reference frame"
python python/generate_rois/generate_rois.py \
${MODEL_MASK_DIR}/${LABEL,,}_2_transformed_mask.nii.gz ${BONE} \
${ATLAS_REG_DIR}/${LABEL,,}_1_atlas_mask_transformed.nii.gz \
${ROI_MASK_DIR} ${LABEL,,}_2_baseline -ow
python python/generate_rois/generate_rois.py \
${MODEL_MASK_DIR}/${LABEL,,}_3_transformed_mask.nii.gz ${BONE} \
${ATLAS_REG_DIR}/${LABEL,,}_1_atlas_mask_transformed.nii.gz \
${ROI_MASK_DIR} ${LABEL,,}_3_baseline -ow
echo "Step 3: Transform the ROIs to the followup reference frame"
if [ ${BONE} = "femur" ]; then
  for ROI_CODE in 10 11 12 13 14 15 16 17;
  do
    blRegistrationApplyTransform \
    ${ROI_MASK_DIR}/${LABEL,,}_2_baseline_roi${ROI_CODE}_mask.nii.gz \
    ${REG_DIR}/${LABEL,,}_2_transform.txt \
    ${ROI_MASK_DIR}/${LABEL,,}_2_roi${ROI_CODE}_mask.nii.gz \
    -fi ${MODEL_MASK_DIR}/${LABEL,,}_2_postprocessed_mask.nii.gz \
    -int NearestNeighbour -it -ow
    blRegistrationApplyTransform \
    ${ROI_MASK_DIR}/${LABEL,,}_3_baseline_roi${ROI_CODE}_mask.nii.gz \
    ${REG_DIR}/${LABEL,,}_3_transform.txt \
    ${ROI_MASK_DIR}/${LABEL,,}_3_roi${ROI_CODE}_mask.nii.gz \
    -fi ${MODEL_MASK_DIR}/${LABEL,,}_3_postprocessed_mask.nii.gz \
    -int NearestNeighbour -it -ow
  done
fi
if [ ${BONE} = "tibia" ]; then
  for ROI_CODE in 30 31 32 33 34 35 36 37;
  do
    blRegistrationApplyTransform \
    ${ROI_MASK_DIR}/${LABEL,,}_2_baseline_roi${ROI_CODE}_mask.nii.gz \
    ${REG_DIR}/${LABEL,,}_2_transform.txt \
    ${ROI_MASK_DIR}/${LABEL,,}_2_roi${ROI_CODE}_mask.nii.gz \
    -fi ${MODEL_MASK_DIR}/${LABEL,,}_2_postprocessed_mask.nii.gz \
    -int NearestNeighbour -it -ow
    blRegistrationApplyTransform \
    ${ROI_MASK_DIR}/${LABEL,,}_3_baseline_roi${ROI_CODE}_mask.nii.gz \
    ${REG_DIR}/${LABEL,,}_3_transform.txt \
    ${ROI_MASK_DIR}/${LABEL,,}_3_roi${ROI_CODE}_mask.nii.gz \
    -fi ${MODEL_MASK_DIR}/${LABEL,,}_3_postprocessed_mask.nii.gz \
    -int NearestNeighbour -it -ow
  done
fi