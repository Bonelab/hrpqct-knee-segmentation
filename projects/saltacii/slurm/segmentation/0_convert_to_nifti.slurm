#!/bin/bash
####### Reserve computing resources #############
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=2:00:00
#SBATCH --mem=128G
#SBATCH --job-name=0_CONVERT_TO_NII
#SBATCH --mail-user=njneetes@ucalgary.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

####### Set environment variables ###############
export PATH="$HOME/software/miniconda3/bin:$PATH"
export PYTHONUNBUFFERED=1
export AIM_DIR="/home/njneetes/work/data/SALTACII/aims"
export NIFTI_DIR="/home/njneetes/work/data/SALTACII/niftis"

####### Run your script #########################
source activate blptl_11.1
python python/aim_nifti/convert_aims_to_nifti.py ${AIM_DIR}/${IMAGE}.AIM ${NIFTI_DIR} -ow