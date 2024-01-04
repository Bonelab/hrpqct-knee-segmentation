#!/bin/bash

# ENVIRONMENT VARIABLES
# these could be set outside of the script, but they are defined here so you know which ones are needed
export PYTHONUNBUFFERED=1
# this just makes python print to the console immediately instead of buffering it, useful for debugging or
# monitoring jobs if you are writing console output to a log, but not necessary if you aren't monitoring
# jobs in real time
export TRAINED_MODEL_DIR="/working_directory/trained_models"
export AIM_DIR="/working_directory/aims"
export NIFTI_DIR="/working_directory/niftis"
export MODEL_MASK_DIR="/working_directory/model_masks"
export OUTPUT_DIR="/working_directory/output"
export ATLAS_REG_DIR="/working_directory/atlas_registrations"
export ATLAS_DIR="/working_directory/atlases/knee/periarticular"

# INPUTS
# here is a list of the inputs that you need to define for this script to run
IMAGE=$1 # the name of the image in all caps (to match the incoming AIM) with no extension
SIDE=$2 # the side of the knee, either "left" or "right"
# NOTE: it's possible that this could be inferred somehow automatically from the AIM processing log
BONE=$3 # the bone of interest, either "tibia" or "femur"
# NOTE: we would probably want one script for femur and one for tibia on the VMS

# COMMANDS
# (0) convert the AIM to a NIFTI
python python/aim_nifti/convert_aims_to_nifti.py ${AIM_DIR}/${IMAGE}.AIM ${NIFTI_DIR} --overwrite
# (1) perform inference on the image
# NOTE: the arguments "hparams-filenames" and "checkpoint-filenames" can be changed, they just have to point to the
# correct files. If you want to organize the trained model hparam and checkpoint files in a different way that would
# be fine. Currently they are all sitting in the log directories where they were saved during training and I have
# not bothered to move or reorganize them.
python python/inference/inference_ensemble.py \
${NIFTI_DIR}/${IMAGE,,}.nii.gz \
${MODEL_MASK_DIR} ${IMAGE,,} \
--hparams-filenames \
${TRAINED_MODEL_DIR}/unetpp_3d_final/version_21096525/ref_hparams.yaml \
${TRAINED_MODEL_DIR}/unetpp_3d_final/version_21096525/ref_hparams.yaml \
${TRAINED_MODEL_DIR}/unetpp_3d_final/version_21096525/ref_hparams.yaml \
${TRAINED_MODEL_DIR}/unetpp_3d_final/version_21096525/ref_hparams.yaml \
${TRAINED_MODEL_DIR}/unetpp_3d_final/version_21096525/ref_hparams.yaml \
--checkpoint-filenames \
${TRAINED_MODEL_DIR}/unetpp_3d_knee_wback_e100_transfer_cv/1701820901_f0/checkpoints/epoch\=151-step\=2888.ckpt \
${TRAINED_MODEL_DIR}/unetpp_3d_knee_wback_e100_transfer_cv/1701820901_f1/checkpoints/epoch\=144-step\=2755.ckpt \
${TRAINED_MODEL_DIR}/unetpp_3d_knee_wback_e100_transfer_cv/1701820901_f2/checkpoints/epoch\=184-step\=3515.ckpt \
${TRAINED_MODEL_DIR}/unetpp_3d_knee_wback_e100_transfer_cv/1701820901_f3/checkpoints/epoch\=180-step\=3439.ckpt \
${TRAINED_MODEL_DIR}/unetpp_3d_knee_wback_e100_transfer_cv/1701820901_f4/checkpoints/epoch\=120-step\=2299.ckpt \
--model-types unet unet unet unet unet \
--patch-width 128 --overlap 0.25 --batch-size 2 \
--overwrite --cuda
# (2) postprocess the segmentation
# NOTE: This script can have an optional "--detect-tunnel" argument if you want the post-processing to try to detect
# a tunnel that would be present if the knee if post ACL reconstruction surgery. But if there is no surgical tunnel
# then you don't want to detect it. It's possible that we would want two different scripts in the VMS, one for each
# case? Not sure how else it could be handled.
python python/postprocessing/postprocess_segmentation.py \
${MODEL_MASK_DIR}/${IMAGE,,}_ensemble_inference_mask.nii.gz \
${MODEL_MASK_DIR} ${IMAGE,,} --overwrite
# (3) save the cort, trab, (and tunnel?) masks as AIMs
# NOTE: the string added to the processing log could be expanded on, something to think about
python python/aim_nifti/convert_masks_to_aims.py \
${MODEL_MASK_DIR}/${IMAGE,,}_postprocessed_mask.nii.gz \
${AIM_DIR}/${IMAGE}.AIM \
${OUTPUT_DIR}/${IMAGE} \
--class-values 1 2 \
--class-labels CORT_MASK TRAB_MASK \
--log "Generated using the code at: https://github.com/Bonelab/hrpqct-knee-segmentation" \
--overwrite
# (4) atlas registration
# (4.1) mask the image using the post-processed bone mask
python python/preprocessing/mask_image.py \
${NIFTI_DIR}/${IMAGE,,}.nii.gz \
${MODEL_MASK_DIR}/${IMAGE,,}_postprocessed_mask.nii.gz \
${NIFTI_DIR}/${IMAGE,,}_masked.nii.gz \
--dilate-amount 35 --background-class 0 --background-value -1000 \
--overwrite
# (4.2) if the image is from a LEFT knee, need to mirror it
if [ ${SIDE} = "left" ]; then
  blImageMirror ${NIFTI_DIR}/${IMAGE,,}_masked.nii.gz 0 ${NIFTI_DIR}/${IMAGE,,}_masked.nii.gz -ow
fi
# (4.3) register the nifti to the atlas
blRegistrationDemons \
${NIFTI_DIR}/${IMAGE,,}_masked.nii.gz ${ATLAS_DIR}/${BONE}/atlas.nii \
${ATLAS_REG_DIR}/${IMAGE,,}_atlas_transform.nii.gz \
-mida -dsf 8 -dss 0.5 -ci Geometry -dt diffeomorphic -mi 200 -ds 2 -us 2 -sf 16 8 4 2 -ss 8 4 2 1 -pmh -ow
# (4.4) transform the atlas mask to the image
blRegistrationApplyTransform \
${ATLAS_DIR}/${BONE}/atlas_mask.nii.gz ${ATLAS_REG_DIR}/${IMAGE,,}_atlas_transform.nii.gz \
${ATLAS_REG_DIR}/${IMAGE,,}_atlas_mask_transformed.nii.gz \
--fixed-image ${NIFTI_DIR}/${IMAGE,,}_masked.nii.gz -int NearestNeighbour -ow
# (4.5) if the image is from a LEFT knee, need to mirror the transformed mask back
if [ ${SIDE} = "left" ]; then
  blImageMirror \
  ${ATLAS_REG_DIR}/${IMAGE,,}_atlas_mask_transformed.nii.gz \
  0 \
  ${ATLAS_REG_DIR}/${IMAGE,,}_atlas_mask_transformed.nii.gz \
  -int NearestNeighbour -ow
fi
# (4.6) remove the masked image
rm ${NIFTI_DIR}/${IMAGE,,}_masked.nii.gz
# (5) generate the ROIs
python python/generate_rois/generate_rois.py \
${MODEL_MASK_DIR}/${IMAGE,,}_postprocessed_mask.nii.gz ${BONE} \
${ATLAS_REG_DIR}/${IMAGE,,}_atlas_mask_transformed.nii.gz \
${OUTPUT_DIR} ${IMAGE,,} --overwrite
# (6) convert the ROIs to AIMs
if [ ${BONE} = "femur" ]; then
  for ROI_CODE in 10 11 12 13 14 15 16 17;
  do
    python python/aim_nifti/convert_mask_to_aim.py \
    ${OUTPUT_DIR}/${IMAGE,,}_roi${ROI_CODE}_mask.nii.gz \
    ${AIM_DIR}/${IMAGE}.AIM \
    ${OUTPUT_DIR}/${IMAGE}_ROI${ROI_CODE}_MASK.AIM \
    -l "Generated using the code at: https://github.com/Bonelab/hrpqct-knee-segmentation" -ow
  done
fi
if [ ${BONE} = "tibia" ]; then
  for ROI_CODE in 30 31 32 33 34 35 36 37;
  do
    python python/aim_nifti/convert_mask_to_aim.py \
    ${OUTPUT_DIR}/${IMAGE,,}_roi${ROI_CODE}_mask.nii.gz \
    ${AIM_DIR}/${IMAGE}.AIM \
    ${OUTPUT_DIR}/${IMAGE}_ROI${ROI_CODE}_MASK.AIM \
    -l "Generated using the code at: https://github.com/Bonelab/hrpqct-knee-segmentation" -ow
  done
fi
# (7) generate the visualizations
source activate blptl_11.1
python python/visualization/write_panning_video.py \
${NIFTI_DIR}/${IMAGE,,}.nii.gz \
${OUTPUT_DIR}/${IMAGE,,}_allrois_mask.nii.gz \
${VIZ_DIR}/${IMAGE,,}_rois \
-ib -400 1400 -pd 1 -ri -cens 10
python python/visualization/write_panning_video.py \
${NIFTI_DIR}/${IMAGE,,}.nii.gz \
${MODEL_MASK_DIR}/${IMAGE,,}_ensemble_inference_mask.nii.gz \
${VIZ_DIR}/${IMAGE,,}_inference \
-ib -400 1400 -pd 1 -ri -cens 10
python python/visualization/write_panning_video.py \
${NIFTI_DIR}/${IMAGE,,}.nii.gz \
${MODEL_MASK_DIR}/${IMAGE,,}_postprocessed_mask.nii.gz \
${VIZ_DIR}/${IMAGE,,}_postprocessed \
-ib -400 1400 -pd 1 -ri -cens 10



