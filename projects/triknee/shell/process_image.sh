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
jid=$(sbatch --export=IMAGE=${IMAGE},BONE=${BONE},SIDE=${SIDE} projects/triknee/slurm/0_preinference.slurm | tr -dc "0-9")
jid=$(sbatch --export=IMAGE=${IMAGE},BONE=${BONE},SIDE=${SIDE} --dependency=afterany:${jid} projects/triknee/slurm/1_inference.slurm | tr -dc "0-9")
sbatch --export=IMAGE=${IMAGE},BONE=${BONE},SIDE=${SIDE} --dependency=afterany:${jid} projects/triknee/slurm/2_postinference.slurm