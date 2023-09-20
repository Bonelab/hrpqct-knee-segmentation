#!/bin/bash
####### Reserve computing resources #############
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=6:00:00
#SBATCH --mem=32G
#SBATCH --job-name=Process-Image
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
export ROI_MASK_DIR="/home/njneetes/work/data/TRIKNEE/roi_masks"
export ATLAS_DIR="/home/njneetes/work/data/Atlases/Knee/periarticular"

####### Run your script #########################
source activate blptl_11.1
echo "Step 1: convert the AIM to a nifti"
python python/aim_nifti/convert_aims_to_nifti.py ${AIM_DIR}/${IMAGE}.AIM ${NIFTI_DIR} -ow
echo ""
echo "Step 2: if the image is from a LEFT knee, need to mirror it"
if [ ${LEFT} ]; then
  echo "Mirroring the image"
  blImageMirror ${NIFTI_DIR}/${IMAGE,,}.nii.gz 0 ${NIFTI_DIR}/${IMAGE,,}.nii.gz -ow
else
  echo "Not mirroring the image"
fi
blImageMirror ${NIFTI_DIR}/${IMAGE,,}.nii.gz 0 ${NIFTI_DIR}/${IMAGE,,}.nii.gz -ow
echo ""
echo "Step 3: register the nifti to the atlas"
blRegistrationDemons \
${NIFTI_DIR}/${IMAGE,,}.nii.gz ${ATLAS_DIR}/${BONE}/atlas.nii \
${ATLAS_REG_DIR}/${IMAGE,,}_atlas_transform.nii.gz \
-mida -dsf 8 -dss 0.5 -ci Geometry -dt diffeomorphic -mi 200 -ds 2 -us 2 -sf 16 8 4 2 -ss 8 4 2 1 -pmh
echo ""
echo "Step 4: transform the atlas mask to the image"
blRegistrationApplyTransform \
${ATLAS_DIR}/${BONE}/atlas_mask.nii.gz ${ATLAS_REG_DIR}/${IMAGE,,}_atlas_transform.nii.gz \
${ATLAS_REG_DIR}/${IMAGE,,}_atlas_mask_transformed.nii.gz \
--fixed-image ${NIFTI_DIR}/${IMAGE,,}.nii.gz -int NearestNeighbour
echo ""
echo "Step 5: if the image is from a LEFT knee, need to mirror the image and transformed mask back"
if [ ${LEFT} ]; then
  echo "Mirroring the image and transformed mask"
  blImageMirror ${NIFTI_DIR}/${IMAGE,,}.nii.gz 0 ${NIFTI_DIR}/${IMAGE,,}.nii.gz -ow
  blImageMirror ${ATLAS_REG_DIR}/${IMAGE,,}_atlas_mask_transformed.nii.gz 0 ${ATLAS_REG_DIR}/${IMAGE,,}_atlas_mask_transformed.nii.gz -ow
else
  echo "Not mirroring the image and transformed mask"
fi
echo ""
echo "Step 6: generate the peri-articular ROIs"
python python/generate_rois/generate_rois.py \
${NIFTI_DIR}/${IMAGE,,}.nii.gz ${BONE} ${ATLAS_REG_DIR}/${IMAGE,,}_atlas_mask_transformed.nii.gz \
${ROI_MASK_DIR}/ ${IMAGE,,} \
-hf /home/njneetes/Projects/hrpqct-knee-segmentation/logs/segresnetvae_3d_final/version_21096522/ref_hparams.yaml \
/home/njneetes/Projects/hrpqct-knee-segmentation/logs/segresnetvae_3d_final/version_21096522/ref_hparams.yaml \
/home/njneetes/Projects/hrpqct-knee-segmentation/logs/segresnetvae_3d_final/version_21096522/ref_hparams.yaml \
/home/njneetes/Projects/hrpqct-knee-segmentation/logs/segresnetvae_3d_final/version_21096522/ref_hparams.yaml \
/home/njneetes/Projects/hrpqct-knee-segmentation/logs/segresnetvae_3d_final/version_21096522/ref_hparams.yaml \
-cf /home/njneetes/Projects/hrpqct-knee-segmentation/logs/segresnetvae_3d_knee_transfer_cv/21115374_f0/checkpoints/epoch\=135-step\=272.ckpt \
/home/njneetes/Projects/hrpqct-knee-segmentation/logs/segresnetvae_3d_knee_transfer_cv/21115374_f1/checkpoints/epoch\=276-step\=554.ckpt \
/home/njneetes/Projects/hrpqct-knee-segmentation/logs/segresnetvae_3d_knee_transfer_cv/21115374_f2/checkpoints/epoch\=197-step\=396.ckpt \
/home/njneetes/Projects/hrpqct-knee-segmentation/logs/segresnetvae_3d_knee_transfer_cv/21115374_f3/checkpoints/epoch\=119-step\=240.ckpt \
/home/njneetes/Projects/hrpqct-knee-segmentation/logs/segresnetvae_3d_knee_transfer_cv/21115374_f4/checkpoints/epoch\=231-step\=464.ckpt \
-mt segresnetvae segresnetvae segresnetvae segresnetvae segresnetvae -adf 20 -pw 64 -ow
echo ""
echo "Step 7: convert roi mask niftis to AIMs"
for i in ${ROI_MASK_DIR}/${IMAGE,,}_*.nii.gz; do
    python python/aim_nifti/convert_mask_to_aim.py ${i} ${AIM_DIR}/${IMAGE}.AIM ${i%.nii.gz}.AIM \
    -l "Generated using the code at: https://github.com/Bonelab/hrpqct-knee-segmentation"
done

