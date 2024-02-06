#!/bin/bash
####### Reserve computing resources #############
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=8:00:00
#SBATCH --mem=150G
#SBATCH --job-name=5_GEN_ROIS
#SBATCH --mail-user=njneetes@ucalgary.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

####### Set environment variables ###############
export PATH="$HOME/software/miniconda3/bin:$PATH"
export PYTHONUNBUFFERED=1
export ATLAS_REG_DIR="/home/njneetes/work/data/SALTACII/atlas_registrations"
export REG_DIR="/home/njneetes/work/data/SALTACII/registrations"
export ROI_MASK_DIR="/home/njneetes/work/data/SALTACII/roi_masks"
export MODEL_MASK_DIR="/home/njneetes/work/data/SALTACII/model_masks"

####### Run your script #########################
source activate blptl_11.1
echo "Step 1: Transform the followup image segmentation to the baseline reference frame"
blRegistrationApplyTransform \
${MODEL_MASK_DIR}/${LABEL,,}_${T1,,}_postprocessed_mask.nii.gz \
${REG_DIR}/${LABEL,,}_${T1,,}_transform.txt \
${REG_DIR}/${LABEL,,}_${T1,,}_bone_mask_baseline.nii.gz \
-fi ${MODEL_MASK_DIR}/${LABEL,,}_${T0,,}_postprocessed_mask.nii.gz \
-int NearestNeighbour -ow
echo "Step 2: Transform the followup image atlas masks to the baseline reference frame"
blRegistrationApplyTransform \
${ATLAS_REG_DIR}/${LABEL,,}_${T1,,}_atlas_mask_transformed.nii.gz \
${REG_DIR}/${LABEL,,}_${T1,,}_transform.txt \
${REG_DIR}/${LABEL,,}_${T1,,}_atlas_mask_baseline.nii.gz \
-fi ${MODEL_MASK_DIR}/${LABEL,,}_${T0,,}_postprocessed_mask.nii.gz \
-int NearestNeighbour -ow
echo "Step 3: Find the intersection of the atlas masks in the baseline reference frame"
python python/postprocessing/intersect_masks.py \
-i \
${ATLAS_REG_DIR}/${LABEL,,}_${T0,,}_atlas_mask_transformed.nii.gz \
${REG_DIR}/${LABEL,,}_${T1,,}_atlas_mask_baseline.nii.gz \
-o ${REG_DIR}/${LABEL,,}_atlas_masks_overlapped_baseline.nii.gz \
-c 1 2 -ow
echo "Step 4: Generate the ROIs in the baseline reference frame with the overlapped atlas masks"
python python/generate_rois/generate_rois.py \
${MODEL_MASK_DIR}/${LABEL,,}_${T0,,}_postprocessed_mask.nii.gz ${BONE} \
${REG_DIR}/${LABEL,,}_atlas_masks_overlapped_baseline.nii.gz \
${ROI_MASK_DIR} ${LABEL,,}_${T0,,} --axial-dilation-footprint 40 -ow
python python/generate_rois/generate_rois.py \
${REG_DIR}/${LABEL,,}_${T1,,}_bone_mask_baseline.nii.gz ${BONE} \
${REG_DIR}/${LABEL,,}_atlas_masks_overlapped_baseline.nii.gz \
${ROI_MASK_DIR} ${LABEL,,}_${T1,,}_baseline --axial-dilation-footprint 40 -ow
echo "Step 5: Transform the followup ROIs to the followup reference frames"
blRegistrationApplyTransform \
${ROI_MASK_DIR}/${LABEL,,}_${T1,,}_baseline_allrois_mask.nii.gz \
${REG_DIR}/${LABEL,,}_${T1,,}_transform.txt \
${ROI_MASK_DIR}/${LABEL,,}_${T1,,}_allrois_mask.nii.gz \
-fi ${MODEL_MASK_DIR}/${LABEL,,}_${T1,,}_postprocessed_mask.nii.gz \
-int NearestNeighbour -it -ow
if [ ${BONE} = "femur" ]; then
  for ROI_CODE in 10 11 12 13 14 15 16 17;
  do
    blRegistrationApplyTransform \
    ${ROI_MASK_DIR}/${LABEL,,}_${T1,,}_baseline_roi${ROI_CODE}_mask.nii.gz \
    ${REG_DIR}/${LABEL,,}_${T1,,}_transform.txt \
    ${ROI_MASK_DIR}/${LABEL,,}_${T1,,}_roi${ROI_CODE}_mask.nii.gz \
    -fi ${MODEL_MASK_DIR}/${LABEL,,}_${T1,,}_postprocessed_mask.nii.gz \
    -int NearestNeighbour -it -ow
  done
fi
if [ ${BONE} = "tibia" ]; then
  for ROI_CODE in 30 31 32 33 34 35 36 37;
  do
    blRegistrationApplyTransform \
    ${ROI_MASK_DIR}/${LABEL,,}_${T1,,}_baseline_roi${ROI_CODE}_mask.nii.gz \
    ${REG_DIR}/${LABEL,,}_${T1,,}_transform.txt \
    ${ROI_MASK_DIR}/${LABEL,,}_${T1,,}_roi${ROI_CODE}_mask.nii.gz \
    -fi ${MODEL_MASK_DIR}/${LABEL,,}_${T1,,}_postprocessed_mask.nii.gz \
    -int NearestNeighbour -it -ow
  done
fi