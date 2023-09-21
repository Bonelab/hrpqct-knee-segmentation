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
JID_PRE=$(sbatch --export=IMAGE=${IMAGE},BONE=${BONE},SIDE=${SIDE} projects/triknee/slurm/0_preinference.slurm | tr -dc "0-9")
echo "Submitted job ${JID_PRE} to perform pre-inference processing."
sleep 1
JID_INF=$(sbatch --export=IMAGE=${IMAGE},BONE=${BONE},SIDE=${SIDE} --dependency=afterany:${JID_PRE} projects/triknee/slurm/1_inference.slurm | tr -dc "0-9")
echo "Submitted job ${JID_INF} to perform inference processing. Will not execute until job ${JID_PRE} is complete."
sleep 1
JID_POST=$(sbatch --export=IMAGE=${IMAGE},BONE=${BONE},SIDE=${SIDE} --dependency=afterany:${JID_INF} projects/triknee/slurm/2_postinference.slurm | tr -dc "0-9")
echo "Submitted job ${JID_POST} to perform post-inference processing. Will not execute until job ${JID_INF} is complete."
sleep 1