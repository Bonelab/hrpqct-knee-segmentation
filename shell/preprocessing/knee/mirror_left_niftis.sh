#!/bin/bash
# Mirror the images

tibia_dir="/home/njneetes/work/data/SALTAC/visit_1/tibia/niftis/"
femur_dir="/home/njneetes/work/data/SALTAC/visit_1/femur/niftis/"


tibias=(
  "SLTC002"
  "SLTC004"
  "SLTC005"
  "SLTC006"
  "SLTC007"
  "SLTC013"
  "SLTC014"
  "SLTC015"
  "SLTC019"
  "SLTC023"
  "SLTC030"
  "SLTC034"
  "SLTC037"
  "SLTC038"
  "SLTC039"
  "SLTC040"
  "SLTC041"
  "SLTC045"
  "SLTC047"
)

femurs=(
  "SLTC002"
  "SLTC004"
  "SLTC005"
  "SLTC006"
  "SLTC007"
  "SLTC013"
  "SLTC014"
  "SLTC015"
  "SLTC019"
  "SLTC023"
  "SLTC030"
  "SLTC034"
  "SLTC037"
  "SLTC038"
  "SLTC039"
  "SLTC040"
  "SLTC041"
  "SLTC045"
  "SLTC047"
)

ax=0

for i in "${tibias[@]}"
do
  echo "Mirroring ${i}"
  img="${tibia_dir}${i}L_T.nii.gz"
  img_mirror="${tibia_dir}${i}LM_T.nii.gz"
  sbatch --export=IMG="$img",AX="$ax",IMG_MIRROR="$img_mirror" slurm/preprocessing/knee/mirror_image.slurm
  sleep 1
done

for i in "${femurs[@]}"
do
  echo "Mirroring ${i}"
  img="${femur_dir}${i}L_F.nii.gz"
  img_mirror="${femur_dir}${i}LM_F.nii.gz"
  sbatch --export=IMG="$img",AX="$ax",IMG_MIRROR="$img_mirror" slurm/preprocessing/knee/mirror_image.slurm
  sleep 1
done
