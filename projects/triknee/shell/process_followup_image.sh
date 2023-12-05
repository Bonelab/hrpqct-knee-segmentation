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
JID_REG=$(sbatch --export=LABEL=${LABEL} projects/triknee/slurm/followup/4_longitudinal_registration.slurm | tr -dc "0-9")
echo "Submitted job ${JID_REG} to register the followup image to the baseline image."