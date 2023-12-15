#!/bin/bash
####### Reserve computing resources #############
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=2:00:00
#SBATCH --mem=64G
#SBATCH --job-name=4_ATLAS_REG
#SBATCH --mail-user=njneetes@ucalgary.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

####### Set environment variables ###############
export PATH="$HOME/software/miniconda3/bin:$PATH"
export PYTHONUNBUFFERED=1
export NIFTI_DIR="/home/njneetes/work/data/PREOA/niftis"
export MODEL_MASK_DIR="/home/njneetes/work/data/PREOA/model_masks"
export ATLAS_REG_DIR="/home/njneetes/work/data/PREOA/atlas_registrations"
export ATLAS_DIR="/home/njneetes/work/data/Atlases/Knee/periarticular"

####### Run your script #########################
source activate blptl_11.1
echo "Step 1: mask the image with the postprocessed model mask"
python python/preprocessing/mask_image.py \
${NIFTI_DIR}/${IMAGE,,}.nii.gz \
${MODEL_MASK_DIR}/${IMAGE,,}_postprocessed_mask.nii.gz \
${NIFTI_DIR}/${IMAGE,,}_masked.nii.gz \
--dilate-amount 35 --background-class 0 --background-value -1000 -ow
echo "Step 2: if the image is from a LEFT knee, need to mirror it"
if [ ${SIDE} = "left" ]; then
  echo "Mirroring the image"
  blImageMirror ${NIFTI_DIR}/${IMAGE,,}_masked.nii.gz 0 ${NIFTI_DIR}/${IMAGE,,}_masked.nii.gz -ow
else
  echo "Not mirroring the image"
fi
echo ""
echo "Step 3: register the nifti to the atlas"
blRegistrationDemons \
${NIFTI_DIR}/${IMAGE,,}_masked.nii.gz ${ATLAS_DIR}/${BONE}/atlas.nii \
${ATLAS_REG_DIR}/${IMAGE,,}_atlas_transform.nii.gz \
-mida -dsf 8 -dss 0.5 -ci Geometry -dt diffeomorphic -mi 200 -ds 2 -us 2 -sf 16 8 4 2 -ss 8 4 2 1 -pmh -ow
echo ""
echo "Step 4: transform the atlas mask to the image"
blRegistrationApplyTransform \
${ATLAS_DIR}/${BONE}/atlas_mask.nii.gz ${ATLAS_REG_DIR}/${IMAGE,,}_atlas_transform.nii.gz \
${ATLAS_REG_DIR}/${IMAGE,,}_atlas_mask_transformed.nii.gz \
--fixed-image ${NIFTI_DIR}/${IMAGE,,}_masked.nii.gz -int NearestNeighbour -ow
echo ""
echo "Step 5: if the image is from a LEFT knee, need to mirror the image and transformed mask back"
if [ ${SIDE} = "left" ]; then
  echo "Mirroring the transformed mask"
  blImageMirror \
  ${ATLAS_REG_DIR}/${IMAGE,,}_atlas_mask_transformed.nii.gz \
  0 \
  ${ATLAS_REG_DIR}/${IMAGE,,}_atlas_mask_transformed.nii.gz \
  -int NearestNeighbour -ow
else
  echo "Not mirroring the image and transformed mask"
fi
echo "remove the masked image"
rm ${NIFTI_DIR}/${IMAGE,,}_masked.nii.gz