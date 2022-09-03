#!/bin/bash
####### Reserve computing resources #############
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --job-name=Preprocess_Validate
#SBATCH --mail-user=njneetes@ucalgary.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

####### Set environment variables ###############
export PATH="$HOME/software/miniconda3/bin:$PATH"

####### Run your script #########################
source activate blptl
python -u python/preprocess_2d_slice_samples.py \
/home/njneetes/work/data/NORMXTII_and_HIPFX/Validate \
/home/njneetes/work/data/NORMXTII_and_HIPFX/Validate_pickled \
-e 50 -is $IS -ie $IE
