#!/bin/bash
####### Reserve computing resources #############
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --mem=150G
#SBATCH --job-name=Preprocess_Train
#SBATCH --mail-user=njneetes@ucalgary.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

####### Set environment variables ###############
export PATH="$HOME/software/miniconda3/bin:$PATH"

####### Activate virtual environment ############
source activate blptl_11.1

####### Run your script #########################
python -u python/misc/extract_periarticular_roi.py /home/njneetes/work/data/SALTAC/visit_"$VISIT_NUM"/"$BONE"