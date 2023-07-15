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
export SCRATCH_DIR="/home/njneetes/work/scratch/nathan/atlas_cv"
export PYTHONUNBUFFERED=1

####### Run your script #########################
source activate blptl_11.1
# STEP 1: construct an atlas using all but one image
python python/atlas/generate_affine_atlas.py \
/scratch/"${SLURM_JOB_ID}"/atlas.nii \
/scratch/"${SLURM_JOB_ID}"/atlas_mask.nii.gz \
-img \
${ATLAS_IMAGES} \
-msk \
${ATLAS_MASKS} \
-dss 0.5 -dsf 8 -ow -ss 8 4 2 1 -sf 16 8 4 2 \
-mai 100 -opt Powell -sm Correlation -smss Regular -smsr 0.3 \
-ds 2 -us 2 -dt diffeomorphic -mdi 200
# STEP 2: register that one image to the atlas
blRegistrationDemons \
"$VALIDATION_IMAGE" \
/scratch/"${SLURM_JOB_ID}"/atlas.nii \
/scratch/"${SLURM_JOB_ID}"/transform.nii \
-ow -mida -dsf 8 -dss 0.5 -ci Geometry -dt diffeomorphic \
-bv -400 -mi 200 -ds 2 -us 2 -sf 16 8 4 2 -ss 8 4 2 1 -pmh
# STEP 3: transform the atlas mask to that one image
blRegistrationApplyTransform \
/scratch/"${SLURM_JOB_ID}"/atlas_mask.nii.gz \
/scratch/"${SLURM_JOB_ID}"/transform.nii \
/scratch/"${SLURM_JOB_ID}"/transformed_mask.nii.gz \
-fi "$VALIDATION_IMAGE" \
-ow -int NearestNeighbour
# STEP 4: combine the ALL_ROI mask into a MED_LAT mask for the one image
python python/atlas/combine_roi_masks.py "$VALIDATION_MASK" /scratch/"${SLURM_JOB_ID}"/reference_mask.nii.gz
# STEP 5: compute overlap metrics
blImageComputeOverlap \
/scratch/"${SLURM_JOB_ID}"/transformed_mask.nii.gz \
/scratch/"${SLURM_JOB_ID}"/reference_mask.nii.gz \
"$SCRATCH_DIR"/"${SLURM_JOB_ID}".csv \
-cl 1 2
# STEP 6: move the yaml files so we have records of what was done to get the corresponding metrics
mv /scratch/"${SLURM_JOB_ID}"/atlas.yaml "$SCRATCH_DIR"/"${SLURM_JOB_ID}"_atlas_generation.yaml
mv /scratch/"${SLURM_JOB_ID}"/transform.yaml "$SCRATCH_DIR"/"${SLURM_JOB_ID}"_registration.yaml
