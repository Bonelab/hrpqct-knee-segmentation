#!/bin/bash
####### Reserve computing resources #############
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=2:00:00
#SBATCH --mem=32G
#SBATCH --job-name=Pre-inference
#SBATCH --mail-user=njneetes@ucalgary.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

####### Set environment variables ###############
export PATH="$HOME/software/miniconda3/bin:$PATH"
export PYTHONUNBUFFERED=1
export AIM_DIR="/home/njneetes/work/data/TRIKNEE/aims"
export NIFTI_DIR="/home/njneetes/work/data/TRIKNEE/niftis"
export ATLAS_REG_DIR="/home/njneetes/work/data/TRIKNEE/atlas_registrations"
export ATLAS_DIR="/home/njneetes/work/data/Atlases/Knee/periarticular"

####### Run your script #########################
source activate blptl_11.1
echo "Step 1: convert the AIM to a nifti"
python python/aim_nifti/convert_aims_to_nifti.py ${AIM_DIR}/${IMAGE}.AIM ${NIFTI_DIR} -ow
echo ""
echo "Step 2: if the image is from a LEFT knee, need to mirror it"
if [ ${SIDE} = "left" ]; then
  echo "Mirroring the image"
  blImageMirror ${NIFTI_DIR}/${IMAGE,,}.nii.gz 0 ${NIFTI_DIR}/${IMAGE,,}.nii.gz -ow
else
  echo "Not mirroring the image"
fi
echo ""
echo "Step 3: register the nifti to the atlas"
blRegistrationDemons \
${NIFTI_DIR}/${IMAGE,,}.nii.gz ${ATLAS_DIR}/${BONE}/atlas.nii \
${ATLAS_REG_DIR}/${IMAGE,,}_atlas_transform.nii.gz \
-mida -dsf 8 -dss 0.5 -ci Geometry -dt diffeomorphic -mi 200 -ds 2 -us 2 -sf 16 8 4 2 -ss 8 4 2 1 -pmh -ow
echo ""
echo "Step 4: transform the atlas mask to the image"
blRegistrationApplyTransform \
${ATLAS_DIR}/${BONE}/atlas_mask.nii.gz ${ATLAS_REG_DIR}/${IMAGE,,}_atlas_transform.nii.gz \
${ATLAS_REG_DIR}/${IMAGE,,}_atlas_mask_transformed.nii.gz \
--fixed-image ${NIFTI_DIR}/${IMAGE,,}.nii.gz -int NearestNeighbour -ow
echo ""
echo "Step 5: if the image is from a LEFT knee, need to mirror the image and transformed mask back"
if [ ${SIDE} = "left" ]; then
  echo "Mirroring the image and transformed mask"
  blImageMirror ${NIFTI_DIR}/${IMAGE,,}.nii.gz 0 ${NIFTI_DIR}/${IMAGE,,}.nii.gz -ow
  blImageMirror \
  ${ATLAS_REG_DIR}/${IMAGE,,}_atlas_mask_transformed.nii.gz \
  0 \
  ${ATLAS_REG_DIR}/${IMAGE,,}_atlas_mask_transformed.nii.gz \
  -int NearestNeighbour -ow
else
  echo "Not mirroring the image and transformed mask"
fi