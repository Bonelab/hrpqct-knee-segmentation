#!/bin/bash
# Mirror the images

tibia_dir="/home/njneetes/work/data/SALTAC/visit_1/tibia/"
femur_dir="/home/njneetes/work/data/SALTAC/visit_1/femur/"


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

for i in "${tibias[@]}"
do
  echo "Mirroring ${i}"
  blImageMirror "${tibia_dir}${i}L_T.nii.gz" "0" "${tibia_dir}${i}LM_T.nii.gz"
done

for i in "${femurs[@]}"
do
  echo "Mirroring ${i}"
  blImageMirror "${femur_dir}${i}L_F.nii.gz" "0" "${femur_dir}${i}LM_F.nii.gz"
done


