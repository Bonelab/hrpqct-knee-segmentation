#!/bin/bash
if [ $# -lt 1 ]
then
  echo "Error: not enough arguments given."
  echo "Required arguments:"
  echo "--------------------"
  echo "Argument 1: name of the nifti file to visualize (with ROI masks), in /work/boyd_lab/data/TRIKNEE/niftis"
  echo "--------------------"
  echo "Example usage: ./projects/conmed/shell/visualize.sh TRIKNEE_01_TL_1"
  echo ""
  exit
fi
IMAGE=$1
sbatch --export=IMAGE=${IMAGE} projects/conmed/slurm/5_visualize.slurm