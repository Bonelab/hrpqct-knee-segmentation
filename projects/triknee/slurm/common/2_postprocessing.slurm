#!/bin/bash
####### Reserve computing resources #############
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=2:00:00
#SBATCH --mem=150G
#SBATCH --job-name=Post-inference
#SBATCH --mail-user=njneetes@ucalgary.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

####### Set environment variables ###############
export PATH="$HOME/software/miniconda3/bin:$PATH"
export PYTHONUNBUFFERED=1
export ROI_MASK_DIR="/home/njneetes/work/data/TRIKNEE/roi_masks_noreg"

####### Run your script #########################
source activate blptl_11.1
python python/postprocessing/postprocess_segmentation.py \
${MODEL_MASK_DIR}/${IMAGE,,}_ensemble_inference_mask.nii.gz \
${MODEL_MASK_DIR} ${IMAGE,,} -ow