#!/bin/bash
if [ $# -lt 2 ]
then
  echo "Error: not enough arguments given."
  echo "Required arguments:"
  echo "--------------------"
  echo "Argument 1: name of the AIM file to process, in /work/boyd_lab/data/PREOA/aims"
  echo "Argument 2: suffix for the first timepoint"
  echo "Argument 3: suffix for the second timepoint"
  echo "Argument 4: the bone, either \`tibia\` or \`femur\`"
  echo "Argument 5: the side, either \`left\` or \`right\`"
  echo "--------------------"
  echo "Example usage: ./projects/saltacii/shell/generate_rois_one_year.sh SLTCII_0002_FR V01 V02 femur right"
  echo ""
  exit
fi
LABEL=$1
T0=$2
T1=$3
BONE=$4
SIDE=$5
JID_ATLT0=$(sbatch --export=IMAGE=${LABEL}_${T0},BONE=${BONE},SIDE=${SIDE} projects/saltacii/slurm/generate_rois_one_year/0_atlas_registration.slurm | tr -dc "0-9")
echo "Submitted job ${JID_ATLT0} to register baseline to atlas."
sleep 0.1
JID_ATLT1=$(sbatch --export=IMAGE=${LABEL}_${T1},BONE=${BONE},SIDE=${SIDE} projects/saltacii/slurm/generate_rois_one_year/0_atlas_registration.slurm | tr -dc "0-9")
echo "Submitted job ${JID_ATLT1} to register followup to atlas."
sleep 0.1
JID_REG=$(sbatch --export=LABEL=${LABEL},T0=${T0},T1=${T1} projects/saltacii/slurm/generate_rois_one_year/1_longitudinal_registration.slurm | tr -dc "0-9")
echo "Submitted job ${JID_REG} to register the followup image to the baseline image."
sleep 0.1
JID_GR=$(sbatch --export=LABEL=${LABEL},T0=${T0},T1=${T1},BONE=${BONE} --dependency=afterany:${JID_ATLT0}:${JID_ATLT1}:${JID_REG} projects/saltacii/slurm/generate_rois_one_year/2_generate_rois.slurm | tr -dc "0-9")
echo "Submitted job ${JID_GR} to generate all ROIs. Will not run until jobs ${JID_ATLT0}, ${JID_ATLT1}, and ${JID_REG} are complete."
sleep 0.1
if [ ${BONE} = "femur" ]; then
  for ROI_CODE in 10 11 12 13 14 15 16 17;
  do
    JID_ROI=$(sbatch --export=IMAGE=${LABEL}_${T0},ROI_CODE=${ROI_CODE} --dependency=afterany:${JID_GR} projects/saltacii/slurm/generate_rois_one_year/3_convert_roi_to_aim.slurm | tr -dc "0-9")
    echo "Submitted job ${JID_ROI} to convert the ROI${ROI_CODE} mask to AIM format. Will not execute until job ${JID_GR} is complete."
    sleep 0.1
    JID_ROI=$(sbatch --export=IMAGE=${LABEL}_${T1},ROI_CODE=${ROI_CODE} --dependency=afterany:${JID_GR} projects/saltacii/slurm/generate_rois_one_year/3_convert_roi_to_aim.slurm | tr -dc "0-9")
    echo "Submitted job ${JID_ROI} to convert the ROI${ROI_CODE} mask to AIM format. Will not execute until job ${JID_GR} is complete."
    sleep 0.1
  done
fi
if [ ${BONE} = "tibia" ]; then
  for ROI_CODE in 30 31 32 33 34 35 36 37;
  do
    JID_ROI=$(sbatch --export=IMAGE=${LABEL}_${T0},ROI_CODE=${ROI_CODE} --dependency=afterany:${JID_GR} projects/saltacii/slurm/generate_rois_one_year/3_convert_roi_to_aim.slurm | tr -dc "0-9")
    echo "Submitted job ${JID_ROI} to convert the ROI${ROI_CODE} mask to AIM format. Will not execute until job ${JID_GR} is complete."
    sleep 0.1
    JID_ROI=$(sbatch --export=IMAGE=${LABEL}_${T1},ROI_CODE=${ROI_CODE} --dependency=afterany:${JID_GR} projects/saltacii/slurm/generate_rois_one_year/3_convert_roi_to_aim.slurm | tr -dc "0-9")
    echo "Submitted job ${JID_ROI} to convert the ROI${ROI_CODE} mask to AIM format. Will not execute until job ${JID_GR} is complete."
    sleep 0.1
  done
fi
sbatch --export=IMAGE=${LABEL}_${T0} --dependency=afterany:${JID_GR} projects/saltacii/slurm/generate_rois_one_year/4_visualize.slurm
echo "Submitted job to generate a visualization. Will not execute until job ${JID_GR} is complete."
sbatch --export=IMAGE=${LABEL}_${T1} --dependency=afterany:${JID_GR} projects/saltacii/slurm/generate_rois_one_year/4_visualize.slurm
echo "Submitted job to generate a visualization. Will not execute until job ${JID_GR} is complete."




