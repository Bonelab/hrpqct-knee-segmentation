#!/bin/bash
####### Reserve computing resources #############
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=4:00:00
#SBATCH --mem=150G
#SBATCH --partition=gpu-v100 --gres=gpu:1
#SBATCH --job-name=1_INFERENCE
#SBATCH --mail-user=njneetes@ucalgary.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

####### Set environment variables ###############
export PATH="$HOME/software/miniconda3/bin:$PATH"
export PYTHONUNBUFFERED=1
export NIFTI_DIR="/home/njneetes/work/data/PREOA/niftis"
export MODEL_MASK_DIR="/home/njneetes/work/data/PREOA/model_masks"

####### Run your script #########################
source activate blptl_11.1
echo "Perform inference on the image"
python python/inference/inference_ensemble.py \
${NIFTI_DIR}/${IMAGE,,}.nii.gz \
${MODEL_MASK_DIR} ${IMAGE,,} \
-hf \
/home/njneetes/Projects/hrpqct-knee-segmentation/from_groot/logs/unetpp_3d_final/version_21096525/ref_hparams.yaml \
/home/njneetes/Projects/hrpqct-knee-segmentation/from_groot/logs/unetpp_3d_final/version_21096525/ref_hparams.yaml \
/home/njneetes/Projects/hrpqct-knee-segmentation/from_groot/logs/unetpp_3d_final/version_21096525/ref_hparams.yaml \
/home/njneetes/Projects/hrpqct-knee-segmentation/from_groot/logs/unetpp_3d_final/version_21096525/ref_hparams.yaml \
/home/njneetes/Projects/hrpqct-knee-segmentation/from_groot/logs/unetpp_3d_final/version_21096525/ref_hparams.yaml \
-cf \
/home/njneetes/Projects/hrpqct-knee-segmentation/from_groot/logs/unetpp_3d_knee_wback_e100_transfer_cv/1701820901_f0/checkpoints/epoch\=151-step\=2888.ckpt \
/home/njneetes/Projects/hrpqct-knee-segmentation/from_groot/logs/unetpp_3d_knee_wback_e100_transfer_cv/1701820901_f1/checkpoints/epoch\=144-step\=2755.ckpt \
/home/njneetes/Projects/hrpqct-knee-segmentation/from_groot/logs/unetpp_3d_knee_wback_e100_transfer_cv/1701820901_f2/checkpoints/epoch\=184-step\=3515.ckpt \
/home/njneetes/Projects/hrpqct-knee-segmentation/from_groot/logs/unetpp_3d_knee_wback_e100_transfer_cv/1701820901_f3/checkpoints/epoch\=180-step\=3439.ckpt \
/home/njneetes/Projects/hrpqct-knee-segmentation/from_groot/logs/unetpp_3d_knee_wback_e100_transfer_cv/1701820901_f4/checkpoints/epoch\=120-step\=2299.ckpt \
-mt unet unet unet unet unet \
-pw 128 -bs 2 -ow --cuda -o 0.25