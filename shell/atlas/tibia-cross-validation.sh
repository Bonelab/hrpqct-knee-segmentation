#!/bin/bash

images=(
  "/home/njneetes/work/data/SALTAC/visit_1/tibia/niftis/SLTC002R_T.nii.gz"
  "/home/njneetes/work/data/SALTAC/visit_1/tibia/niftis/SLTC004R_T.nii.gz"
  "/home/njneetes/work/data/SALTAC/visit_1/tibia/niftis/SLTC005R_T.nii.gz"
  "/home/njneetes/work/data/SALTAC/visit_1/tibia/niftis/SLTC006R_T.nii.gz"
  "/home/njneetes/work/data/SALTAC/visit_1/tibia/niftis/SLTC007R_T.nii.gz"
  "/home/njneetes/work/data/SALTAC/visit_1/tibia/niftis/SLTC013R_T.nii.gz"
  "/home/njneetes/work/data/SALTAC/visit_1/tibia/niftis/SLTC014R_T.nii.gz"
  "/home/njneetes/work/data/SALTAC/visit_1/tibia/niftis/SLTC015R_T.nii.gz"
  "/home/njneetes/work/data/SALTAC/visit_1/tibia/niftis/SLTC019R_T.nii.gz"
  "/home/njneetes/work/data/SALTAC/visit_1/tibia/niftis/SLTC023R_T.nii.gz"
  "/home/njneetes/work/data/SALTAC/visit_1/tibia/niftis/SLTC030R_T.nii.gz"
  "/home/njneetes/work/data/SALTAC/visit_1/tibia/niftis/SLTC034R_T.nii.gz"
  "/home/njneetes/work/data/SALTAC/visit_1/tibia/niftis/SLTC037R_T.nii.gz"
  "/home/njneetes/work/data/SALTAC/visit_1/tibia/niftis/SLTC038R_T.nii.gz"
  "/home/njneetes/work/data/SALTAC/visit_1/tibia/niftis/SLTC039R_T.nii.gz"
  "/home/njneetes/work/data/SALTAC/visit_1/tibia/niftis/SLTC040R_T.nii.gz"
  "/home/njneetes/work/data/SALTAC/visit_1/tibia/niftis/SLTC041R_T.nii.gz"
  "/home/njneetes/work/data/SALTAC/visit_1/tibia/niftis/SLTC045R_T.nii.gz"
  "/home/njneetes/work/data/SALTAC/visit_1/tibia/niftis/SLTC047R_T.nii.gz"
  "/home/njneetes/work/data/SALTAC/visit_1/tibia/niftis/SLTC002LM_T.nii.gz"
  "/home/njneetes/work/data/SALTAC/visit_1/tibia/niftis/SLTC004LM_T.nii.gz"
  "/home/njneetes/work/data/SALTAC/visit_1/tibia/niftis/SLTC005LM_T.nii.gz"
  "/home/njneetes/work/data/SALTAC/visit_1/tibia/niftis/SLTC006LM_T.nii.gz"
  "/home/njneetes/work/data/SALTAC/visit_1/tibia/niftis/SLTC007LM_T.nii.gz"
  "/home/njneetes/work/data/SALTAC/visit_1/tibia/niftis/SLTC013LM_T.nii.gz"
  "/home/njneetes/work/data/SALTAC/visit_1/tibia/niftis/SLTC014LM_T.nii.gz"
  "/home/njneetes/work/data/SALTAC/visit_1/tibia/niftis/SLTC015LM_T.nii.gz"
  "/home/njneetes/work/data/SALTAC/visit_1/tibia/niftis/SLTC019LM_T.nii.gz"
  "/home/njneetes/work/data/SALTAC/visit_1/tibia/niftis/SLTC023LM_T.nii.gz"
  "/home/njneetes/work/data/SALTAC/visit_1/tibia/niftis/SLTC030LM_T.nii.gz"
  "/home/njneetes/work/data/SALTAC/visit_1/tibia/niftis/SLTC034LM_T.nii.gz"
  "/home/njneetes/work/data/SALTAC/visit_1/tibia/niftis/SLTC037LM_T.nii.gz"
  "/home/njneetes/work/data/SALTAC/visit_1/tibia/niftis/SLTC038LM_T.nii.gz"
  "/home/njneetes/work/data/SALTAC/visit_1/tibia/niftis/SLTC039LM_T.nii.gz"
  "/home/njneetes/work/data/SALTAC/visit_1/tibia/niftis/SLTC040LM_T.nii.gz"
  "/home/njneetes/work/data/SALTAC/visit_1/tibia/niftis/SLTC041LM_T.nii.gz"
  "/home/njneetes/work/data/SALTAC/visit_1/tibia/niftis/SLTC045LM_T.nii.gz"
  "/home/njneetes/work/data/SALTAC/visit_1/tibia/niftis/SLTC047LM_T.nii.gz"
)

masks=(
  "/home/njneetes/work/data/SALTAC/visit_1/tibia/niftis/SLTC002R_T_ROI_ALL.nii.gz"
  "/home/njneetes/work/data/SALTAC/visit_1/tibia/niftis/SLTC004R_T_ROI_ALL.nii.gz"
  "/home/njneetes/work/data/SALTAC/visit_1/tibia/niftis/SLTC005R_T_ROI_ALL.nii.gz"
  "/home/njneetes/work/data/SALTAC/visit_1/tibia/niftis/SLTC006R_T_ROI_ALL.nii.gz"
  "/home/njneetes/work/data/SALTAC/visit_1/tibia/niftis/SLTC007R_T_ROI_ALL.nii.gz"
  "/home/njneetes/work/data/SALTAC/visit_1/tibia/niftis/SLTC013R_T_ROI_ALL.nii.gz"
  "/home/njneetes/work/data/SALTAC/visit_1/tibia/niftis/SLTC014R_T_ROI_ALL.nii.gz"
  "/home/njneetes/work/data/SALTAC/visit_1/tibia/niftis/SLTC015R_T_ROI_ALL.nii.gz"
  "/home/njneetes/work/data/SALTAC/visit_1/tibia/niftis/SLTC019R_T_ROI_ALL.nii.gz"
  "/home/njneetes/work/data/SALTAC/visit_1/tibia/niftis/SLTC023R_T_ROI_ALL.nii.gz"
  "/home/njneetes/work/data/SALTAC/visit_1/tibia/niftis/SLTC030R_T_ROI_ALL.nii.gz"
  "/home/njneetes/work/data/SALTAC/visit_1/tibia/niftis/SLTC034R_T_ROI_ALL.nii.gz"
  "/home/njneetes/work/data/SALTAC/visit_1/tibia/niftis/SLTC037R_T_ROI_ALL.nii.gz"
  "/home/njneetes/work/data/SALTAC/visit_1/tibia/niftis/SLTC038R_T_ROI_ALL.nii.gz"
  "/home/njneetes/work/data/SALTAC/visit_1/tibia/niftis/SLTC039R_T_ROI_ALL.nii.gz"
  "/home/njneetes/work/data/SALTAC/visit_1/tibia/niftis/SLTC040R_T_ROI_ALL.nii.gz"
  "/home/njneetes/work/data/SALTAC/visit_1/tibia/niftis/SLTC041R_T_ROI_ALL.nii.gz"
  "/home/njneetes/work/data/SALTAC/visit_1/tibia/niftis/SLTC045R_T_ROI_ALL.nii.gz"
  "/home/njneetes/work/data/SALTAC/visit_1/tibia/niftis/SLTC047R_T_ROI_ALL.nii.gz"
  "/home/njneetes/work/data/SALTAC/visit_1/tibia/niftis/SLTC002LM_T_ROI_ALL.nii.gz"
  "/home/njneetes/work/data/SALTAC/visit_1/tibia/niftis/SLTC004LM_T_ROI_ALL.nii.gz"
  "/home/njneetes/work/data/SALTAC/visit_1/tibia/niftis/SLTC005LM_T_ROI_ALL.nii.gz"
  "/home/njneetes/work/data/SALTAC/visit_1/tibia/niftis/SLTC006LM_T_ROI_ALL.nii.gz"
  "/home/njneetes/work/data/SALTAC/visit_1/tibia/niftis/SLTC007LM_T_ROI_ALL.nii.gz"
  "/home/njneetes/work/data/SALTAC/visit_1/tibia/niftis/SLTC013LM_T_ROI_ALL.nii.gz"
  "/home/njneetes/work/data/SALTAC/visit_1/tibia/niftis/SLTC014LM_T_ROI_ALL.nii.gz"
  "/home/njneetes/work/data/SALTAC/visit_1/tibia/niftis/SLTC015LM_T_ROI_ALL.nii.gz"
  "/home/njneetes/work/data/SALTAC/visit_1/tibia/niftis/SLTC019LM_T_ROI_ALL.nii.gz"
  "/home/njneetes/work/data/SALTAC/visit_1/tibia/niftis/SLTC023LM_T_ROI_ALL.nii.gz"
  "/home/njneetes/work/data/SALTAC/visit_1/tibia/niftis/SLTC030LM_T_ROI_ALL.nii.gz"
  "/home/njneetes/work/data/SALTAC/visit_1/tibia/niftis/SLTC034LM_T_ROI_ALL.nii.gz"
  "/home/njneetes/work/data/SALTAC/visit_1/tibia/niftis/SLTC037LM_T_ROI_ALL.nii.gz"
  "/home/njneetes/work/data/SALTAC/visit_1/tibia/niftis/SLTC038LM_T_ROI_ALL.nii.gz"
  "/home/njneetes/work/data/SALTAC/visit_1/tibia/niftis/SLTC039LM_T_ROI_ALL.nii.gz"
  "/home/njneetes/work/data/SALTAC/visit_1/tibia/niftis/SLTC040LM_T_ROI_ALL.nii.gz"
  "/home/njneetes/work/data/SALTAC/visit_1/tibia/niftis/SLTC041LM_T_ROI_ALL.nii.gz"
  "/home/njneetes/work/data/SALTAC/visit_1/tibia/niftis/SLTC045LM_T_ROI_ALL.nii.gz"
  "/home/njneetes/work/data/SALTAC/visit_1/tibia/niftis/SLTC047LM_T_ROI_ALL.nii.gz"
)

echo "Number of images: ${#images[@]}"
echo "Number of masks: ${#masks[@]}"
echo ""
echo ""


for (( i=0; i<${#images[@]}; i++ ))
do
  validation_image=${images[$i]}
  validation_mask=${masks[$i]}
  atlas_images=("${images[@]:0:$i}" "${images[@]:$((i+1)):${#images[@]}}")
  atlas_masks=("${masks[@]:0:$i}" "${masks[@]:$((i+1)):${#images[@]}}")
  echo "======================"
  echo "Index: $i"
  echo "---------"
  echo "Validation Image:"
  echo "$validation_image"
  echo ""
  echo "Validation Mask:"
  echo "$validation_mask"
  echo ""
  echo "Atlas Images:"
  echo "${atlas_images[@]}"
  echo ""
  echo "Atlas Masks:"
  echo "${atlas_masks[@]}"
  echo ""
  echo "---------"
  sbatch \
  --export=ATLAS_IMAGES="${atlas_images[*]}",ATLAS_MASKS="${atlas_masks[*]}",VALIDATION_IMAGE="$validation_image",VALIDATION_MASK="$validation_mask" \
  slurm/atlas/cross-validation.slurm
  sleep 1
  echo "======================"
done
