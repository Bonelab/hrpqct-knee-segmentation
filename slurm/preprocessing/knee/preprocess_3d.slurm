#!/bin/bash
####### Reserve computing resources #############
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --job-name=Preprocess_Train
#SBATCH --mail-user=njneetes@ucalgary.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

####### Set environment variables ###############
export PATH="$HOME/software/miniconda3/bin:$PATH"

####### Activate virtual environment ############
source activate blptl

####### Run your script #########################
python -u python/preprocessing/preprocess_3d_patch_samples_from_npz.py \
/home/njneetes/work/data/"$STUDY"/visit_"$VISIT_NUM"/"$BONE"/"$COMPARTMENT" \
/home/njneetes/work/data/"$STUDY"/visit_"$VISIT_NUM"/preprocessed/"$BONE"_"$COMPARTMENT"_pickled_3d \
--patch-width 64 -e 100