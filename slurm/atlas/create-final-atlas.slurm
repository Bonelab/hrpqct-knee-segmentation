#!/bin/bash
####### Reserve computing resources #############
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --job-name=Atlas-CV
#SBATCH --mail-user=njneetes@ucalgary.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

####### Set environment variables ###############
export PATH="$HOME/software/miniconda3/bin:$PATH"
export PYTHONUNBUFFERED=1

####### Run your script #########################
source activate blptl_11.1
python python/atlas/generate_affine_atlas.py \
${ATLAS_DIR}/atlas.nii \
${ATLAS_DIR}/atlas_mask.nii.gz \
-img \
${ATLAS_IMAGES} \
-msk \
${ATLAS_MASKS} \
-dss 0.5 -dsf 8 -ow -ss 8 4 2 1 -sf 16 8 4 2 \
-mai 100 -opt Powell -sm Correlation -smss Regular -smsr 0.3 \
-ds 2 -us 2 -dt diffeomorphic -mdi 200