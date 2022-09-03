#!/bin/bash
####### Reserve computing resources #############
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --time=8:00:00
#SBATCH --mem=200G
#SBATCH --partition=gpu-v100 --gres=gpu:1
#SBATCH --job-name=Train_UNET2D
#SBATCH --mail-user=njneetes@ucalgary.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

####### Set environment variables ###############
export PATH="$HOME/software/miniconda3/bin:$PATH"

####### Run your script #########################
source activate blptl
python -u python/train_unet_2d_pickled.py -td /home/njneetes/Data/HRpQCT/Train_pickled -vd /home/njneetes/Data/HRpQCT/Validate_pickled -bs 64 -e 100 -w 4 -ng 1 -l unet2d -v ${SLURM_JOB_ID}
