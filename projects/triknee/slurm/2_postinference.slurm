#!/bin/bash
####### Reserve computing resources #############
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=2:00:00
#SBATCH --mem=150G
#SBATCH --job-name=Post-inference
#SBATCH --mail-user=njneetes@ucalgary.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

####### Set environment variables ###############
export PATH="$HOME/software/miniconda3/bin:$PATH"
export PYTHONUNBUFFERED=1
export NIFTI_DIR="/home/njneetes/work/data/TRIKNEE/niftis"
export ATLAS_REG_DIR="/home/njneetes/work/data/TRIKNEE/atlas_registrations"
export ROI_MASK_DIR="/home/njneetes/work/data/TRIKNEE/roi_masks_noreg"
export MODEL_MASK_DIR="/home/njneetes/work/data/TRIKNEE/model_masks"
export VIZ_DIR="/home/njneetes/work/data/TRIKNEE/visualizations"

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
echo "Step 3: generate a visualization to check ROI quality"
python python/visualization/write_panning_video.py \
${NIFTI_DIR}/${IMAGE,,}.nii.gz \
${ROI_MASK_DIR}/${IMAGE,,}_allrois_mask.nii.gz \
${VIZ_DIR}/${IMAGE,,} \
-ib -400 1400 -pd 1 -ri -cens 10