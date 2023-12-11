#!/bin/bash
####### Reserve computing resources #############
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=4:00:00
#SBATCH --mem=100G
#SBATCH --partition=gpu-v100 --gres=gpu:1
#SBATCH --job-name=1_INFERENCE
#SBATCH --mail-user=njneetes@ucalgary.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

####### Set environment variables ###############
export PATH="$HOME/software/miniconda3/bin:$PATH"
export PYTHONUNBUFFERED=1
export NIFTI_DIR="/home/njneetes/work/data/CONMED/niftis"
export MODEL_MASK_DIR="/home/njneetes/work/data/CONMED/model_masks"

####### Run your script #########################
source activate blptl_11.1
echo "Perform inference on the image"
python python/inference/inference_ensemble.py \
${NIFTI_DIR}/${IMAGE,,}.nii.gz \
${MODEL_MASK_DIR} ${IMAGE,,} \
-hf /home/njneetes/Projects/hrpqct-knee-segmentation/logs/unetpp_3d_final/version_21096525/ref_hparams.yaml \
-cf /home/njneetes/Projects/hrpqct-knee-segmentation/logs/unetpp_3d_knee_wback_transfer_final/version_25943069/checkpoints/epoch\=257-step\=6192.ckpt \
-mt unet -pw 128 -bs 2 -ow --cuda -o 0.5