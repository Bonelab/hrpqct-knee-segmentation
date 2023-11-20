#!/bin/bash
####### Reserve computing resources #############
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=2:00:00
#SBATCH --mem=100G
#SBATCH --partition=gpu-v100 --gres=gpu:1
#SBATCH --job-name=Inference
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
-hf \
/home/njneetes/Projects/hrpqct-knee-segmentation/logs/unetpp_3d_final/version_21096525/ref_hparams.yaml \
/home/njneetes/Projects/hrpqct-knee-segmentation/logs/unetpp_3d_final/version_21096525/ref_hparams.yaml \
/home/njneetes/Projects/hrpqct-knee-segmentation/logs/unetpp_3d_final/version_21096525/ref_hparams.yaml \
/home/njneetes/Projects/hrpqct-knee-segmentation/logs/unetpp_3d_final/version_21096525/ref_hparams.yaml \
/home/njneetes/Projects/hrpqct-knee-segmentation/logs/unetpp_3d_final/version_21096525/ref_hparams.yaml \
-cf \
/home/njneetes/Projects/hrpqct-knee-segmentation/logs/unetpp_3d_knee_wback_transfer_cv/24500746_f0/checkpoints/epoch\=125-step\=2394.ckpt \
/home/njneetes/Projects/hrpqct-knee-segmentation/logs/unetpp_3d_knee_wback_transfer_cv/24500746_f1/checkpoints/epoch\=174-step\=3325.ckpt \
/home/njneetes/Projects/hrpqct-knee-segmentation/logs/unetpp_3d_knee_wback_transfer_cv/24500746_f2/checkpoints/epoch\=167-step\=3192.ckpt \
/home/njneetes/Projects/hrpqct-knee-segmentation/logs/unetpp_3d_knee_wback_transfer_cv/24500746_f3/checkpoints/epoch\=189-step\=3610.ckpt \
/home/njneetes/Projects/hrpqct-knee-segmentation/logs/unetpp_3d_knee_wback_transfer_cv/24500746_f4/checkpoints/epoch\=105-step\=2014.ckpt \
-mt unet unet unet unet unet -pw 64 -bs 16 -ow --cuda -o 0.75