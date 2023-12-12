#!/bin/bash
if [ $# -lt 3 ]
then
  echo "Error: not enough arguments given."
  echo "Required arguments:"
  echo "--------------------"
  echo "Argument 1: name of the AIM file to process, in /work/boyd_lab/data/CONMED/aims"
  echo "Argument 2: which bone that image is, either \`tibia\` or \`femur\`"
  echo "Argument 3: which side that image is, either \`left\` or \`right\`"
  echo "--------------------"
  echo "Example usage: ./projects/conmed/shell/process_image.sh CONMD01R_T tibia right"
  echo ""
  exit
fi
IMAGE=$1
BONE=$2
SIDE=$3
#JID_NII=$(sbatch --export=IMAGE=${IMAGE},BONE=${BONE},SIDE=${SIDE} projects/conmed/slurm/0_convert_to_nifti.slurm | tr -dc "0-9")
#echo "Submitted job ${JID_NII} to convert AIM to nifti."
#sleep 0.1
#JID_INF=$(sbatch --export=IMAGE=${IMAGE},BONE=${BONE},SIDE=${SIDE} --dependency=afterany:${JID_NII} projects/conmed/slurm/1_inference.slurm | tr -dc "0-9")
#echo "Submitted job ${JID_INF} to perform inference processing. Will not execute until job ${JID_NII} is complete."
#sleep 0.1
#JID_PP=$(sbatch --export=IMAGE=${IMAGE},BONE=${BONE},SIDE=${SIDE} --dependency=afterany:${JID_INF} projects/conmed/slurm/2_postprocessing.slurm | tr -dc "0-9")
#echo "Submitted job ${JID_PP} to postprocess segmentation. Will not execute until job ${JID_INF} is complete."
JID_PP=$(sbatch --export=IMAGE=${IMAGE},BONE=${BONE},SIDE=${SIDE} projects/conmed/slurm/2_postprocessing.slurm | tr -dc "0-9")
echo "Submitted job ${JID_PP} to postprocess segmentation."
sleep 0.1
JID_CTA=$(sbatch --export=IMAGE=${IMAGE},BONE=${BONE},SIDE=${SIDE} --dependency=afterany:${JID_PP} projects/conmed/slurm/3_convert_segmentation_to_aims.slurm | tr -dc "0-9")
echo "Submitted job ${JID_CTA} to convert the segmentation to AIMs. Will not execute until job ${JID_PP} is complete."
sleep 0.1
JID_AR=$(sbatch --export=IMAGE=${IMAGE},BONE=${BONE},SIDE=${SIDE} --dependency=afterany:${JID_PP} projects/conmed/slurm/4_atlas_registration.slurm | tr -dc "0-9")
echo "Submitted job ${JID_AR} to register the image to the atlas. Will not execute until job ${JID_PP} is complete."
sleep 0.1
JID_GR=$(sbatch --export=IMAGE=${IMAGE},BONE=${BONE},SIDE=${SIDE} --dependency=afterany:${JID_AR} projects/conmed/slurm/5_generate_rois.slurm | tr -dc "0-9")
echo "Submitted job ${JID_GR} to generate the ROIs. Will not execute until job ${JID_AR} is complete."
sleep 0.1
if [ ${BONE} = "femur" ]; then
  for ROI_CODE in 10 11 12 13 14 15 16 17;
  do
    JID_ROI=$(sbatch --export=IMAGE=${IMAGE},BONE=${BONE},SIDE=${SIDE},ROI_CODE=${ROI_CODE} --dependency=afterany:${JID_GR} projects/conmed/slurm/6_convert_roi_to_aim.slurm | tr -dc "0-9")
    echo "Submitted job ${JID_ROI} to convert the ROI${ROI_CODE} mask to AIM format. Will not execute until job ${JID_GR} is complete."
    sleep 0.1
  done
fi
if [ ${BONE} = "tibia" ]; then
  for ROI_CODE in 30 31 32 33 34 35 36 37;
  do
    JID_ROI=$(sbatch --export=IMAGE=${IMAGE},BONE=${BONE},SIDE=${SIDE},ROI_CODE=${ROI_CODE} --dependency=afterany:${JID_GR} projects/conmed/slurm/6_convert_roi_to_aim.slurm | tr -dc "0-9")
    echo "Submitted job ${JID_ROI} to convert the ROI${ROI_CODE} mask to AIM format. Will not execute until job ${JID_GR} is complete."
    sleep 0.1
  done
fi
sbatch --export=IMAGE=${IMAGE},BONE=${BONE},SIDE=${SIDE} --dependency=afterany:${JID_GR} projects/conmed/slurm/7_visualize.slurm
echo "Submitted job to generate a visualization. Will not execute until job ${JID_GR} is complete."