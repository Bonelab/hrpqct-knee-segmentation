#!/bin/bash
DATA_DIR="/home/njneetes/work/data/KNEE_VALIDATION/femur"

sbatch --export=DATA_DIR="/home/njneetes/work/data/KNEE_VALIDATION/femur",IMAGE="PTFT_002_61" slurm/inference/inference_unet.slurm
sleep 1
sbatch --export=DATA_DIR="/home/njneetes/work/data/KNEE_VALIDATION/femur",IMAGE="PTFT_003_61" slurm/inference/inference_unet.slurm
sleep 1
sbatch --export=DATA_DIR="/home/njneetes/work/data/KNEE_VALIDATION/femur",IMAGE="PTFT_006_61" slurm/inference/inference_unet.slurm
sleep 1
sbatch --export=DATA_DIR="/home/njneetes/work/data/KNEE_VALIDATION/femur",IMAGE="PTFT_008_61" slurm/inference/inference_unet.slurm
sleep 1
sbatch --export=DATA_DIR="/home/njneetes/work/data/KNEE_VALIDATION/femur",IMAGE="PTFT_009_61" slurm/inference/inference_unet.slurm
sleep 1
sbatch --export=DATA_DIR="/home/njneetes/work/data/KNEE_VALIDATION/femur",IMAGE="PTFT_012_61" slurm/inference/inference_unet.slurm
sleep 1
sbatch --export=DATA_DIR="/home/njneetes/work/data/KNEE_VALIDATION/femur",IMAGE="PTFT_013_61" slurm/inference/inference_unet.slurm
sleep 1
sbatch --export=DATA_DIR="/home/njneetes/work/data/KNEE_VALIDATION/femur",IMAGE="PTFT_018_61" slurm/inference/inference_unet.slurm
sleep 1
sbatch --export=DATA_DIR="/home/njneetes/work/data/KNEE_VALIDATION/femur",IMAGE="PTFT_019_61" slurm/inference/inference_unet.slurm
sleep 1
sbatch --export=DATA_DIR="/home/njneetes/work/data/KNEE_VALIDATION/femur",IMAGE="PTFT_020_61" slurm/inference/inference_unet.slurm
sleep 1
sbatch --export=DATA_DIR="/home/njneetes/work/data/KNEE_VALIDATION/femur",IMAGE="PTFT_027_61" slurm/inference/inference_unet.slurm
sleep 1
sbatch --export=DATA_DIR="/home/njneetes/work/data/KNEE_VALIDATION/femur",IMAGE="PTFT_030_61" slurm/inference/inference_unet.slurm
sleep 1
