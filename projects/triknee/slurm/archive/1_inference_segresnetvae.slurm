#!/bin/bash
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
export NIFTI_DIR="/home/njneetes/work/data/TRIKNEE/niftis"
export MODEL_MASK_DIR="/home/njneetes/work/data/TRIKNEE/model_masks"

####### Run your script #########################
source activate blptl_11.1
echo "Perform inference on the image"
python python/inference/inference_ensemble.py \
${NIFTI_DIR}/${IMAGE,,}.nii.gz \
${MODEL_MASK_DIR} ${IMAGE,,} \
-hf /home/njneetes/Projects/hrpqct-knee-segmentation/logs/segresnetvae_3d_final/version_21096522/ref_hparams.yaml \
/home/njneetes/Projects/hrpqct-knee-segmentation/logs/segresnetvae_3d_final/version_21096522/ref_hparams.yaml \
/home/njneetes/Projects/hrpqct-knee-segmentation/logs/segresnetvae_3d_final/version_21096522/ref_hparams.yaml \
/home/njneetes/Projects/hrpqct-knee-segmentation/logs/segresnetvae_3d_final/version_21096522/ref_hparams.yaml \
/home/njneetes/Projects/hrpqct-knee-segmentation/logs/segresnetvae_3d_final/version_21096522/ref_hparams.yaml \
-cf /home/njneetes/Projects/hrpqct-knee-segmentation/logs/segresnetvae_3d_knee_transfer_cv/21115374_f0/checkpoints/epoch\=135-step\=272.ckpt \
/home/njneetes/Projects/hrpqct-knee-segmentation/logs/segresnetvae_3d_knee_transfer_cv/21115374_f1/checkpoints/epoch\=276-step\=554.ckpt \
/home/njneetes/Projects/hrpqct-knee-segmentation/logs/segresnetvae_3d_knee_transfer_cv/21115374_f2/checkpoints/epoch\=197-step\=396.ckpt \
/home/njneetes/Projects/hrpqct-knee-segmentation/logs/segresnetvae_3d_knee_transfer_cv/21115374_f3/checkpoints/epoch\=119-step\=240.ckpt \
/home/njneetes/Projects/hrpqct-knee-segmentation/logs/segresnetvae_3d_knee_transfer_cv/21115374_f4/checkpoints/epoch\=231-step\=464.ckpt \
-mt segresnetvae segresnetvae segresnetvae segresnetvae segresnetvae -pw 64 -bs 16 -ow --cuda -o 0.25