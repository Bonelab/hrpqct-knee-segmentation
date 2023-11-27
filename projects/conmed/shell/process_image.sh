#!/bin/bash
if [ $# -lt 3 ]
then
  echo "Error: not enough arguments given."
  echo "Required arguments:"
  echo "--------------------"
  echo "Argument 1: name of the AIM file to process, in /work/boyd_lab/data/TRIKNEE/aims"
  echo "Argument 2: which bone that image is, either \`tibia\` or \`femur\`"
  echo "Argument 3: which side that image is, either \`left\` or \`right\`"
  echo "--------------------"
  echo "Example usage: ./projects/triknee/shell/process_image.sh TRIKNEE_01_TL_1 tibia left"
  echo ""
  exit
fi
IMAGE=$1
BONE=$2
SIDE=$3
JID_PRE=$(sbatch --export=IMAGE=${IMAGE},BONE=${BONE},SIDE=${SIDE} projects/conmed/slurm/0_preinference.slurm | tr -dc "0-9")
echo "Submitted job ${JID_PRE} to perform pre-inference processing."
sleep 0.1
JID_INF=$(sbatch --export=IMAGE=${IMAGE},BONE=${BONE},SIDE=${SIDE} --dependency=afterany:${JID_PRE} projects/conmed/slurm/1_inference_unetpp_final.slurm | tr -dc "0-9")
echo "Submitted job ${JID_INF} to perform inference processing. Will not execute until job ${JID_PRE} is complete."
sleep 0.1
JID_POST=$(sbatch --export=IMAGE=${IMAGE},BONE=${BONE},SIDE=${SIDE} --dependency=afterany:${JID_INF} projects/conmed/slurm/2_postinference.slurm | tr -dc "0-9")
echo "Submitted job ${JID_POST} to perform post-inference processing. Will not execute until job ${JID_INF} is complete."
sleep 0.1
JID_CTC=$(sbatch --export=IMAGE=${IMAGE},BONE=${BONE},SIDE=${SIDE} --dependency=afterany:${JID_POST} projects/conmed/slurm/4_convert_to_aims.slurm | tr -dc "0-9")
echo "Submitted job ${JID_CTC} to convert the cort/trab/tunnel masks to AIM format. Will not execute until job ${JID_POST} is complete."
if [ ${BONE} = "femur" ]; then
  for ROI_CODE in 10 11 12 13 14 15 16 17;
  do
    JID_ROI=$(sbatch --export=IMAGE=${IMAGE},BONE=${BONE},SIDE=${SIDE},ROI_CODE=${ROI_CODE} --dependency=afterany:${JID_POST} projects/conmed/slurm/3_convert_to_aim.slurm | tr -dc "0-9")
    echo "Submitted job ${JID_ROI} to convert the ROI${ROI_CODE} mask to AIM format. Will not execute until job ${JID_POST} is complete."
    sleep 0.1
  done
fi
if [ ${BONE} = "tibia" ]; then
  for ROI_CODE in 30 31 32 33 34 35 36 37;
  do
    JID_ROI=$(sbatch --export=IMAGE=${IMAGE},BONE=${BONE},SIDE=${SIDE},ROI_CODE=${ROI_CODE} --dependency=afterany:${JID_POST} projects/conmed/slurm/3_convert_to_aim.slurm | tr -dc "0-9")
    echo "Submitted job ${JID_ROI} to convert the ROI${ROI_CODE} mask to AIM format. Will not execute until job ${JID_POST} is complete."
    sleep 0.1
  done
fi