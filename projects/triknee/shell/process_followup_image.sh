#!/bin/bash
if [ $# -lt 2 ]
then
  echo "Error: not enough arguments given."
  echo "Required arguments:"
  echo "--------------------"
  echo "Argument 1: LABEL of the AIM file to process, in /work/boyd_lab/data/TRIKNEE/aims"
  echo "Argument 3: which bone that image is, either \`tibia\` or \`femur\`"
  echo "--------------------"
  echo "Example usage: ./projects/triknee/shell/process_followup_image.sh TRIKNEE_01_TL 2 tibia"
  echo ""
  exit
fi
LABEL=$1
BONE=$2
#JID_REG=$(sbatch --export=LABEL=${LABEL} projects/triknee/slurm/followup/4_longitudinal_registration.slurm | tr -dc "0-9")
#echo "Submitted job ${JID_REG} to register the followup image to the baseline image."
#sleep 0.1
#JID_GR=$(sbatch --export=LABEL=${LABEL},BONE=${BONE} --dependency=afterany:${JID_REG} projects/triknee/slurm/followup/5_generate_rois.slurm | tr -dc "0-9")
#echo "Submitted job ${JID_GR} to generate the ROIs. Will not execute until job ${JID_REG} is complete."
#sleep 0.1
if [ ${BONE} = "femur" ]; then
  for ROI_CODE in 10 11 12 13 14 15 16 17;
  do
    JID_ROI=$(sbatch --export=IMAGE=${LABEL}_1,ROI_CODE=${ROI_CODE} --dependency=afterany:${JID_GR} projects/triknee/slurm/followup/6_convert_roi_to_aim.slurm | tr -dc "0-9")
    echo "Submitted job ${JID_ROI} to convert the ROI${ROI_CODE} mask to AIM format. Will not execute until job ${JID_GR} is complete."
    sleep 0.1
    JID_ROI=$(sbatch --export=IMAGE=${LABEL}_2,ROI_CODE=${ROI_CODE} --dependency=afterany:${JID_GR} projects/triknee/slurm/followup/6_convert_roi_to_aim.slurm | tr -dc "0-9")
    echo "Submitted job ${JID_ROI} to convert the ROI${ROI_CODE} mask to AIM format. Will not execute until job ${JID_GR} is complete."
    sleep 0.1
    JID_ROI=$(sbatch --export=IMAGE=${LABEL}_3,ROI_CODE=${ROI_CODE} --dependency=afterany:${JID_GR} projects/triknee/slurm/followup/6_convert_roi_to_aim.slurm | tr -dc "0-9")
    echo "Submitted job ${JID_ROI} to convert the ROI${ROI_CODE} mask to AIM format. Will not execute until job ${JID_GR} is complete."
    sleep 0.1
  done
fi
if [ ${BONE} = "tibia" ]; then
  for ROI_CODE in 30 31 32 33 34 35 36 37;
  do
    JID_ROI=$(sbatch --export=IMAGE=${LABEL}_1,ROI_CODE=${ROI_CODE} --dependency=afterany:${JID_GR} projects/triknee/slurm/followup/6_convert_roi_to_aim.slurm | tr -dc "0-9")
    echo "Submitted job ${JID_ROI} to convert the ROI${ROI_CODE} mask to AIM format. Will not execute until job ${JID_GR} is complete."
    sleep 0.1
    JID_ROI=$(sbatch --export=IMAGE=${LABEL}_2,ROI_CODE=${ROI_CODE} --dependency=afterany:${JID_GR} projects/triknee/slurm/followup/6_convert_roi_to_aim.slurm | tr -dc "0-9")
    echo "Submitted job ${JID_ROI} to convert the ROI${ROI_CODE} mask to AIM format. Will not execute until job ${JID_GR} is complete."
    sleep 0.1
    JID_ROI=$(sbatch --export=IMAGE=${LABEL}_3,ROI_CODE=${ROI_CODE} --dependency=afterany:${JID_GR} projects/triknee/slurm/followup/6_convert_roi_to_aim.slurm | tr -dc "0-9")
    echo "Submitted job ${JID_ROI} to convert the ROI${ROI_CODE} mask to AIM format. Will not execute until job ${JID_GR} is complete."
    sleep 0.1
  done
fi
sbatch --export=IMAGE=${LABEL}_1 --dependency=afterany:${JID_GR} projects/triknee/slurm/followup/7_visualize.slurm
echo "Submitted job to generate a visualization. Will not execute until job ${JID_GR} is complete."
sbatch --export=IMAGE=${LABEL}_2 --dependency=afterany:${JID_GR} projects/triknee/slurm/followup/7_visualize.slurm
echo "Submitted job to generate a visualization. Will not execute until job ${JID_GR} is complete."
sbatch --export=IMAGE=${LABEL}_3 --dependency=afterany:${JID_GR} projects/triknee/slurm/followup/7_visualize.slurm
echo "Submitted job to generate a visualization. Will not execute until job ${JID_GR} is complete."