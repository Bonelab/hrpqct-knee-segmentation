#!/bin/bash
# Convert all of the AIMS files to NIFTI files

tibia_dir="/home/njneetes/work/data/SALTAC/visit_1/tibia/"
femur_dir="/home/njneetes/work/data/SALTAC/visit_1/femur/"

tibias=(
  "SLTC002L_T_ROI_ALL"
  "SLTC002R_T_ROI_ALL"
  "SLTC004L_T_ROI_ALL"
  "SLTC004R_T_ROI_ALL"
  "SLTC005L_T_ROI_ALL"
  "SLTC005R_T_ROI_ALL"
  "SLTC006L_T_ROI_ALL"
  "SLTC006R_T_ROI_ALL"
  "SLTC007L_T_ROI_ALL"
  "SLTC007R_T_ROI_ALL"
  "SLTC013L_T_ROI_ALL"
  "SLTC013R_T_ROI_ALL"
  "SLTC014L_T_ROI_ALL"
  "SLTC014R_T_ROI_ALL"
  "SLTC015L_T_ROI_ALL"
  "SLTC015R_T_ROI_ALL"
  "SLTC019L_T_ROI_ALL"
  "SLTC019R_T_ROI_ALL"
  "SLTC023L_T_ROI_ALL"
  "SLTC023R_T_ROI_ALL"
  "SLTC030L_T_ROI_ALL"
  "SLTC030R_T_ROI_ALL"
  "SLTC034L_T_ROI_ALL"
  "SLTC034R_T_ROI_ALL"
  "SLTC037L_T_ROI_ALL"
  "SLTC037R_T_ROI_ALL"
  "SLTC038L_T_ROI_ALL"
  "SLTC038R_T_ROI_ALL"
  "SLTC039L_T_ROI_ALL"
  "SLTC039R_T_ROI_ALL"
  "SLTC040L_T_ROI_ALL"
  "SLTC040R_T_ROI_ALL"
  "SLTC041L_T_ROI_ALL"
  "SLTC041R_T_ROI_ALL"
  "SLTC045L_T_ROI_ALL"
  "SLTC045R_T_ROI_ALL"
  "SLTC047L_T_ROI_ALL"
  "SLTC047R_T_ROI_ALL"
)

femurs=(
  "SLTC002L_F_ROI_ALL"
  "SLTC002R_F_ROI_ALL"
  "SLTC004L_F_ROI_ALL"
  "SLTC004R_F_ROI_ALL"
  "SLTC005L_F_ROI_ALL"
  "SLTC005R_F_ROI_ALL"
  "SLTC006L_F_ROI_ALL"
  "SLTC006R_F_ROI_ALL"
  "SLTC007L_F_ROI_ALL"
  "SLTC007R_F_ROI_ALL"
  "SLTC013L_F_ROI_ALL"
  "SLTC013R_F_ROI_ALL"
  "SLTC014L_F_ROI_ALL"
  "SLTC014R_F_ROI_ALL"
  "SLTC015L_F_ROI_ALL"
  "SLTC015R_F_ROI_ALL"
  "SLTC019L_F_ROI_ALL"
  "SLTC019R_F_ROI_ALL"
  "SLTC023L_F_ROI_ALL"
  "SLTC023R_F_ROI_ALL"
  "SLTC030L_F_ROI_ALL"
  "SLTC030R_F_ROI_ALL"
  "SLTC034L_F_ROI_ALL"
  "SLTC034R_F_ROI_ALL"
  "SLTC037L_F_ROI_ALL"
  "SLTC037R_F_ROI_ALL"
  "SLTC038L_F_ROI_ALL"
  "SLTC038R_F_ROI_ALL"
  "SLTC039L_F_ROI_ALL"
  "SLTC039R_F_ROI_ALL"
  "SLTC040L_F_ROI_ALL"
  "SLTC040R_F_ROI_ALL"
  "SLTC041L_F_ROI_ALL"
  "SLTC041R_F_ROI_ALL"
  "SLTC045L_F_ROI_ALL"
  "SLTC045R_F_ROI_ALL"
  "SLTC047L_F_ROI_ALL"
  "SLTC047R_F_ROI_ALL"
)

for tibia in "${tibias[@]}"
do
  aim="${tibia_dir}${tibia}.AIM"
  nii="${tibia_dir}/niftis/${tibia}.nii.gz"
  sbatch --export=AIM="${aim}",NII="${nii}" slurm/preprocessing/knee/convert_aim_mask_to_nifti.slurm
  sleep 1
done

for femur in "${femurs[@]}"
do
  aim="${femur_dir}${femur}.AIM"
  nii="${femur_dir}/niftis/${femur}.nii.gz"
  sbatch --export=AIM="${aim}",NII="${nii}" slurm/preprocessing/knee/convert_aim_mask_to_nifti.slurm
  sleep 1
done