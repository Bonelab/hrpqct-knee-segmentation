#!/bin/bash

images=(
  "/home/njneetes/work/data/SALTAC/visit_1/femur/niftis/SLTC002R_F.nii.gz"
  "/home/njneetes/work/data/SALTAC/visit_1/femur/niftis/SLTC004R_F.nii.gz"
  "/home/njneetes/work/data/SALTAC/visit_1/femur/niftis/SLTC005R_F.nii.gz"
  "/home/njneetes/work/data/SALTAC/visit_1/femur/niftis/SLTC006R_F.nii.gz"
  "/home/njneetes/work/data/SALTAC/visit_1/femur/niftis/SLTC007R_F.nii.gz"
  "/home/njneetes/work/data/SALTAC/visit_1/femur/niftis/SLTC013R_F.nii.gz"
  "/home/njneetes/work/data/SALTAC/visit_1/femur/niftis/SLTC014R_F.nii.gz"
  "/home/njneetes/work/data/SALTAC/visit_1/femur/niftis/SLTC015R_F.nii.gz"
  "/home/njneetes/work/data/SALTAC/visit_1/femur/niftis/SLTC019R_F.nii.gz"
  "/home/njneetes/work/data/SALTAC/visit_1/femur/niftis/SLTC023R_F.nii.gz"
  "/home/njneetes/work/data/SALTAC/visit_1/femur/niftis/SLTC030R_F.nii.gz"
  "/home/njneetes/work/data/SALTAC/visit_1/femur/niftis/SLTC034R_F.nii.gz"
  "/home/njneetes/work/data/SALTAC/visit_1/femur/niftis/SLTC037R_F.nii.gz"
  "/home/njneetes/work/data/SALTAC/visit_1/femur/niftis/SLTC038R_F.nii.gz"
  "/home/njneetes/work/data/SALTAC/visit_1/femur/niftis/SLTC039R_F.nii.gz"
  "/home/njneetes/work/data/SALTAC/visit_1/femur/niftis/SLTC040R_F.nii.gz"
  "/home/njneetes/work/data/SALTAC/visit_1/femur/niftis/SLTC041R_F.nii.gz"
  "/home/njneetes/work/data/SALTAC/visit_1/femur/niftis/SLTC045R_F.nii.gz"
  "/home/njneetes/work/data/SALTAC/visit_1/femur/niftis/SLTC047R_F.nii.gz"
  "/home/njneetes/work/data/SALTAC/visit_1/femur/niftis/SLTC002LM_F.nii.gz"
  "/home/njneetes/work/data/SALTAC/visit_1/femur/niftis/SLTC004LM_F.nii.gz"
  "/home/njneetes/work/data/SALTAC/visit_1/femur/niftis/SLTC005LM_F.nii.gz"
  "/home/njneetes/work/data/SALTAC/visit_1/femur/niftis/SLTC006LM_F.nii.gz"
  "/home/njneetes/work/data/SALTAC/visit_1/femur/niftis/SLTC007LM_F.nii.gz"
  "/home/njneetes/work/data/SALTAC/visit_1/femur/niftis/SLTC013LM_F.nii.gz"
  "/home/njneetes/work/data/SALTAC/visit_1/femur/niftis/SLTC014LM_F.nii.gz"
  "/home/njneetes/work/data/SALTAC/visit_1/femur/niftis/SLTC015LM_F.nii.gz"
  "/home/njneetes/work/data/SALTAC/visit_1/femur/niftis/SLTC019LM_F.nii.gz"
  "/home/njneetes/work/data/SALTAC/visit_1/femur/niftis/SLTC023LM_F.nii.gz"
  "/home/njneetes/work/data/SALTAC/visit_1/femur/niftis/SLTC030LM_F.nii.gz"
  "/home/njneetes/work/data/SALTAC/visit_1/femur/niftis/SLTC034LM_F.nii.gz"
  "/home/njneetes/work/data/SALTAC/visit_1/femur/niftis/SLTC037LM_F.nii.gz"
  "/home/njneetes/work/data/SALTAC/visit_1/femur/niftis/SLTC038LM_F.nii.gz"
  "/home/njneetes/work/data/SALTAC/visit_1/femur/niftis/SLTC039LM_F.nii.gz"
  "/home/njneetes/work/data/SALTAC/visit_1/femur/niftis/SLTC040LM_F.nii.gz"
  "/home/njneetes/work/data/SALTAC/visit_1/femur/niftis/SLTC041LM_F.nii.gz"
  "/home/njneetes/work/data/SALTAC/visit_1/femur/niftis/SLTC045LM_F.nii.gz"
  "/home/njneetes/work/data/SALTAC/visit_1/femur/niftis/SLTC047LM_F.nii.gz"
)

masks=(
  "/home/njneetes/work/data/SALTAC/visit_1/femur/niftis/SLTC002R_F_ROI_ALL.nii.gz"
  "/home/njneetes/work/data/SALTAC/visit_1/femur/niftis/SLTC004R_F_ROI_ALL.nii.gz"
  "/home/njneetes/work/data/SALTAC/visit_1/femur/niftis/SLTC005R_F_ROI_ALL.nii.gz"
  "/home/njneetes/work/data/SALTAC/visit_1/femur/niftis/SLTC006R_F_ROI_ALL.nii.gz"
  "/home/njneetes/work/data/SALTAC/visit_1/femur/niftis/SLTC007R_F_ROI_ALL.nii.gz"
  "/home/njneetes/work/data/SALTAC/visit_1/femur/niftis/SLTC013R_F_ROI_ALL.nii.gz"
  "/home/njneetes/work/data/SALTAC/visit_1/femur/niftis/SLTC014R_F_ROI_ALL.nii.gz"
  "/home/njneetes/work/data/SALTAC/visit_1/femur/niftis/SLTC015R_F_ROI_ALL.nii.gz"
  "/home/njneetes/work/data/SALTAC/visit_1/femur/niftis/SLTC019R_F_ROI_ALL.nii.gz"
  "/home/njneetes/work/data/SALTAC/visit_1/femur/niftis/SLTC023R_F_ROI_ALL.nii.gz"
  "/home/njneetes/work/data/SALTAC/visit_1/femur/niftis/SLTC030R_F_ROI_ALL.nii.gz"
  "/home/njneetes/work/data/SALTAC/visit_1/femur/niftis/SLTC034R_F_ROI_ALL.nii.gz"
  "/home/njneetes/work/data/SALTAC/visit_1/femur/niftis/SLTC037R_F_ROI_ALL.nii.gz"
  "/home/njneetes/work/data/SALTAC/visit_1/femur/niftis/SLTC038R_F_ROI_ALL.nii.gz"
  "/home/njneetes/work/data/SALTAC/visit_1/femur/niftis/SLTC039R_F_ROI_ALL.nii.gz"
  "/home/njneetes/work/data/SALTAC/visit_1/femur/niftis/SLTC040R_F_ROI_ALL.nii.gz"
  "/home/njneetes/work/data/SALTAC/visit_1/femur/niftis/SLTC041R_F_ROI_ALL.nii.gz"
  "/home/njneetes/work/data/SALTAC/visit_1/femur/niftis/SLTC045R_F_ROI_ALL.nii.gz"
  "/home/njneetes/work/data/SALTAC/visit_1/femur/niftis/SLTC047R_F_ROI_ALL.nii.gz"
  "/home/njneetes/work/data/SALTAC/visit_1/femur/niftis/SLTC002LM_F_ROI_ALL.nii.gz"
  "/home/njneetes/work/data/SALTAC/visit_1/femur/niftis/SLTC004LM_F_ROI_ALL.nii.gz"
  "/home/njneetes/work/data/SALTAC/visit_1/femur/niftis/SLTC005LM_F_ROI_ALL.nii.gz"
  "/home/njneetes/work/data/SALTAC/visit_1/femur/niftis/SLTC006LM_F_ROI_ALL.nii.gz"
  "/home/njneetes/work/data/SALTAC/visit_1/femur/niftis/SLTC007LM_F_ROI_ALL.nii.gz"
  "/home/njneetes/work/data/SALTAC/visit_1/femur/niftis/SLTC013LM_F_ROI_ALL.nii.gz"
  "/home/njneetes/work/data/SALTAC/visit_1/femur/niftis/SLTC014LM_F_ROI_ALL.nii.gz"
  "/home/njneetes/work/data/SALTAC/visit_1/femur/niftis/SLTC015LM_F_ROI_ALL.nii.gz"
  "/home/njneetes/work/data/SALTAC/visit_1/femur/niftis/SLTC019LM_F_ROI_ALL.nii.gz"
  "/home/njneetes/work/data/SALTAC/visit_1/femur/niftis/SLTC023LM_F_ROI_ALL.nii.gz"
  "/home/njneetes/work/data/SALTAC/visit_1/femur/niftis/SLTC030LM_F_ROI_ALL.nii.gz"
  "/home/njneetes/work/data/SALTAC/visit_1/femur/niftis/SLTC034LM_F_ROI_ALL.nii.gz"
  "/home/njneetes/work/data/SALTAC/visit_1/femur/niftis/SLTC037LM_F_ROI_ALL.nii.gz"
  "/home/njneetes/work/data/SALTAC/visit_1/femur/niftis/SLTC038LM_F_ROI_ALL.nii.gz"
  "/home/njneetes/work/data/SALTAC/visit_1/femur/niftis/SLTC039LM_F_ROI_ALL.nii.gz"
  "/home/njneetes/work/data/SALTAC/visit_1/femur/niftis/SLTC040LM_F_ROI_ALL.nii.gz"
  "/home/njneetes/work/data/SALTAC/visit_1/femur/niftis/SLTC041LM_F_ROI_ALL.nii.gz"
  "/home/njneetes/work/data/SALTAC/visit_1/femur/niftis/SLTC045LM_F_ROI_ALL.nii.gz"
  "/home/njneetes/work/data/SALTAC/visit_1/femur/niftis/SLTC047LM_F_ROI_ALL.nii.gz"
)

atlas_dir="/home/njneetes/work/data/Atlases/Knee/periarticular/femur"

sbatch \
--export=ATLAS_DIR="${atlas_dir}",ATLAS_IMAGES="${images[*]}",ATLAS_MASKS="${masks[*]}" \
slurm/atlas/create-final-atlas.slurm