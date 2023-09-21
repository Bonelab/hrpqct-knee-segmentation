#!/bin/bash
####### Reserve computing resources #############
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=2:00:00
#SBATCH --mem=100G
#SBATCH --job-name=Post-inference
#SBATCH --mail-user=njneetes@ucalgary.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

####### Set environment variables ###############
export PATH="$HOME/software/miniconda3/bin:$PATH"
export PYTHONUNBUFFERED=1
export AIM_DIR="/home/njneetes/work/data/TRIKNEE/aims"
export ATLAS_REG_DIR="/home/njneetes/work/data/TRIKNEE/atlas_registrations"
export ROI_MASK_DIR="/home/njneetes/work/data/TRIKNEE/roi_masks"
export MODEL_MASK_DIR="/home/njneetes/work/data/TRIKNEE/model_masks"

####### Run your script #########################
source activate blptl_11.1
echo "Step 1: Post-process the bone mask"
python python/postprocessing/postprocess_segmentation.py \
${MODEL_MASK_DIR}/${IMAGE,,}_ensemble_inference_mask.nii.gz \
${MODEL_MASK_DIR} ${IMAGE,,} -ow
echo ""
echo "Step 2: generate the peri-articular ROIs"
python python/generate_rois/generate_rois.py \
${MODEL_MASK_DIR}/${IMAGE,,}_postprocessed_mask.nii.gz ${BONE} \
${ATLAS_REG_DIR}/${IMAGE,,}_atlas_mask_transformed.nii.gz \
${ROI_MASK_DIR} ${IMAGE,,} -ow
echo ""
echo "Step 3: convert roi mask niftis to AIMs"
for ROI_CODE in 10 11 12 13 14 15 16 17 30 31 32 33 34 35 36 37;
do
    python python/aim_nifti/convert_mask_to_aim.py \
    ${IMAGE,,}_roi${ROI_CODE}_mask.nii.gz ${AIM_DIR}/${IMAGE}.AIM ${IMAGE}_ROI${ROI_CODE}_MASK.AIM \
    -l "Generated using the code at: https://github.com/Bonelab/hrpqct-knee-segmentation"
done