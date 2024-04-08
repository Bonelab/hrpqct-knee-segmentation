#!/bin/bash
if [ $# -lt 2 ]
then
  echo "Error: not enough arguments given."
  echo "Required arguments:"
  echo "--------------------"
  echo "Argument 1: name of the AIM file to process, in /work/boyd_lab/data/PREOA/aims"
  echo "Argument 2: whether this image is the injured leg, post-surgery, either \`yes\` or \`no\`"
  echo "--------------------"
  echo "Example usage: ./projects/saltacii/shell/segment_image.sh SLTCII_0002_FR_V01 no"
  echo ""
  exit
fi
IMAGE=$1
SURGERY=$2
if [ ${SURGERY} = "yes" ]; then
  JID_PP=$(sbatch --export=IMAGE=${IMAGE} projects/saltacii/slurm/segmentation/2b_postprocessing_tunnel.slurm | tr -dc "0-9")
  echo "Submitted job ${JID_PP} to postprocess segmentation with tunnel detection. Will not execute until job ${JID_INF} is complete."
  sbatch --export=IMAGE=${IMAGE} --dependency=afterany:${JID_PP} projects/saltacii/slurm/segmentation/3b_convert_segmentation_to_aims_tunnel.slurm
  echo "Submitted job to convert to AIMs. Will not execute until job ${JID_PP} is complete."
else
  JID_PP=$(sbatch --export=IMAGE=${IMAGE} projects/saltacii/slurm/segmentation/2a_postprocessing.slurm | tr -dc "0-9")
  echo "Submitted job ${JID_PP} to postprocess segmentation. Will not execute until job ${JID_INF} is complete."
  sbatch --export=IMAGE=${IMAGE} --dependency=afterany:${JID_PP} projects/saltacii/slurm/segmentation/3a_convert_segmentation_to_aims.slurm
  echo "Submitted job to convert to AIMs. Will not execute until job ${JID_PP} is complete."
fi
sbatch --export=IMAGE=${IMAGE} --dependency=afterany:${JID_PP} projects/saltacii/slurm/segmentation/4_visualize.slurm
echo "Submitted job to generate a visualization. Will not execute until job ${JID_PP} is complete."