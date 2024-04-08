#!/bin/bash
if [ $# -lt 5 ]
then
  echo "Error: not enough arguments given (requires 5)."
fi
IMAGE=$1
COMPARTMENT=$2
REG=$3
ROI_CODE=$5
sbatch \
--export=IMAGE=${IMAGE},COMPARTMENT=${COMPARTMENT},REG=${REG},ROI_CODE=${ROI_CODE} \
projects/triknee/slurm/treece/get_thickness.slurm