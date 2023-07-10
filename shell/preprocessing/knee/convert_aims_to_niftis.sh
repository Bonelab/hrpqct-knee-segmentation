#!/bin/bash
# Convert all of the AIMS files to NIFTI files

tibia_dir="/home/njneetes/work/data/SALTAC/visit_1/tibia/"
femur_dir="/home/njneetes/work/data/SALTAC/visit_1/femur/"

tibias=(
  "SLTC002L_T"
  "SLTC002R_T"
  "SLTC004L_T"
  "SLTC004R_T"
  "SLTC005L_T"
  "SLTC005R_T"
  "SLTC006L_T"
  "SLTC006R_T"
  "SLTC007L_T"
  "SLTC007R_T"
  "SLTC013L_T"
  "SLTC013R_T"
  "SLTC014L_T"
  "SLTC014R_T"
  "SLTC015L_T"
  "SLTC015R_T"
  "SLTC019L_T"
  "SLTC019R_T"
  "SLTC023L_T"
  "SLTC023R_T"
  "SLTC030L_T"
  "SLTC030R_T"
  "SLTC034L_T"
  "SLTC034R_T"
  "SLTC037L_T"
  "SLTC037R_T"
  "SLTC038L_T"
  "SLTC038R_T"
  "SLTC039L_T"
  "SLTC039R_T"
  "SLTC040L_T"
  "SLTC040R_T"
  "SLTC041L_T"
  "SLTC041R_T"
  "SLTC045L_T"
  "SLTC045R_T"
  "SLTC047L_T"
  "SLTC047R_T"
)

femurs=(
  "SLTC002L_F"
  "SLTC002R_F"
  "SLTC004L_F"
  "SLTC004R_F"
  "SLTC005L_F"
  "SLTC005R_F"
  "SLTC006L_F"
  "SLTC006R_F"
  "SLTC007L_F"
  "SLTC007R_F"
  "SLTC013L_F"
  "SLTC013R_F"
  "SLTC014L_F"
  "SLTC014R_F"
  "SLTC015L_F"
  "SLTC015R_F"
  "SLTC019L_F"
  "SLTC019R_F"
  "SLTC023L_F"
  "SLTC023R_F"
  "SLTC030L_F"
  "SLTC030R_F"
  "SLTC034L_F"
  "SLTC034R_F"
  "SLTC037L_F"
  "SLTC037R_F"
  "SLTC038L_F"
  "SLTC038R_F"
  "SLTC039L_F"
  "SLTC039R_F"
  "SLTC040L_F"
  "SLTC040R_F"
  "SLTC041L_F"
  "SLTC041R_F"
  "SLTC045L_F"
  "SLTC045R_F"
  "SLTC047L_F"
  "SLTC047R_F"
)

for tibia in "${tibias[@]}"
do
  aim="${tibia_dir}${tibia}.AIM"
  nii="${tibia_dir}/niftis/${tibia}.nii.gz"
  sbatch --EXPORT=AIM="${aim}",NII="${nii}" slurm/preprocessing/knee/convert_aim_to_nifti.slurm
  sleep 1
done

for femur in "${femurs[@]}"
do
  aim="${femur_dir}${femur}.AIM"
  nii="${femur_dir}/niftis/${femur}.nii.gz"
  sbatch --EXPORT=AIM="${aim}",NII="${nii}" slurm/preprocessing/knee/convert_aim_to_nifti.slurm
  sleep 1
done
