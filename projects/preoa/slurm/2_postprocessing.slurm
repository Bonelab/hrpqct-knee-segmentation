#!/bin/bash
####### Reserve computing resources #############
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=3:00:00
#SBATCH --mem=250G
#SBATCH --job-name=2_POSTPROC
#SBATCH --mail-user=njneetes@ucalgary.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

####### Set environment variables ###############
export PATH="$HOME/software/miniconda3/bin:$PATH"
export PYTHONUNBUFFERED=1
export MODEL_MASK_DIR="/home/njneetes/work/data/PREOA/model_masks"

####### Run your script #########################
source activate blptl_11.1
python python/postprocessing/postprocess_segmentation.py \
${MODEL_MASK_DIR}/${IMAGE,,}_ensemble_inference_mask.nii.gz \
${MODEL_MASK_DIR} ${IMAGE,,} -ow