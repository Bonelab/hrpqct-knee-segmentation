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
JID_NII=$(sbatch --export=IMAGE=${IMAGE} projects/saltacii/slurm/segmentation/0_convert_to_nifti.slurm | tr -dc "0-9")
echo "Submitted job ${JID_NII} to convert AIM to nifti."
sleep 0.1
JID_INF=$(sbatch --export=IMAGE=${IMAGE} --dependency=afterany:${JID_NII} projects/saltacii/slurm/segmentation/1_inference.slurm | tr -dc "0-9")
echo "Submitted job ${JID_INF} to perform inference processing. Will not execute until job ${JID_NII} is complete."
sleep 0.1
JID_PP=$(sbatch --export=IMAGE=${IMAGE} --dependency=afterany:${JID_INF} projects/saltacii/slurm/segmentation/2a_postprocessing.slurm | tr -dc "0-9")
echo "Submitted job ${JID_PP} to postprocess segmentation. Will not execute until job ${JID_INF} is complete."
sleep 0.1
if [ ${SURGERY} = "yes" ]; then
  JID_PP=$(sbatch --export=IMAGE=${IMAGE} projects/saltacii/slurm/segmentation/2b_postprocessing_tunnel.slurm | tr -dc "0-9")
  echo "Submitted job ${JID_PP} to postprocess segmentation with tunnel detection."
else
  JID_PP=$(sbatch --export=IMAGE=${IMAGE} projects/saltacii/slurm/segmentation/2a_postprocessing.slurm | tr -dc "0-9")
  echo "Submitted job ${JID_PP} to postprocess segmentation."
fi
sbatch --export=IMAGE=${IMAGE} --dependency=afterany:${JID_PP} projects/saltacii/slurm/segmentation/4_visualize.slurm
echo "Submitted job to generate a visualization. Will not execute until job ${JID_PP} is complete."