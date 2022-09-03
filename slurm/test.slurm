#!/bin/bash
####### Reserve computing resources #############
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --time=8:00:00
#SBATCH --mem=0
#SBATCH --partition=gpu-v100 --gres=gpu:2
#SBATCH --job-name=UNET2D
#SBATCH --mail-user=njneetes@ucalgary.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

####### Set environment variables ###############
export PATH="~/software/miniconda3/bin:$PATH"

####### Run your script #########################
source activate blptl
python -u train_unet_2d.py -td /home/njneetes/Data/HRpQCT/Train -vd /home/njneetes/Data/HRpQCT/Validate -bs 64 -e 100 -w 4 -pw 160 -ng 2 -l unet2d -v ${SLURM_JOB_ID} -dm
