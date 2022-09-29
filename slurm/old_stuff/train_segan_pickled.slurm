#!/bin/bash
####### Reserve computing resources #############
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=8:00:00
#SBATCH --mem=100G
#SBATCH --partition=gpu-v100 --gres=gpu:1
#SBATCH --job-name=SEGAN_2D
#SBATCH --mail-user=njneetes@ucalgary.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

####### Set environment variables ###############
export PATH="$HOME/software/miniconda3/bin:$PATH"

####### Run your script #########################
source activate blptl
python python/train_segan.py \
-td /home/njneetes/work/data/NORMXTII_and_HIPFX/Train_pickled \
-vd /home/njneetes/work/data/NORMXTII_and_HIPFX/Validate_pickled \
-f 64 128 256 512 1024 \
-bs 64 -e 200 -esp 25 -w 8 -ng 1 -l SeGAN2D_Testing