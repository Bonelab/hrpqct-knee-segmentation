#!/bin/bash
DATA_DIR="/home/njneetes/work/data/SALTAC/visit_1/femur"

sbatch --export=DATA_DIR="$DATA_DIR",IMAGE="SLTC001L_F" slurm/inference/inference_unet.slurm; sleep 1
sbatch --export=DATA_DIR="$DATA_DIR",IMAGE="SLTC001R_F" slurm/inference/inference_unet.slurm; sleep 1
sbatch --export=DATA_DIR="$DATA_DIR",IMAGE="SLTC002L_F" slurm/inference/inference_unet.slurm; sleep 1
sbatch --export=DATA_DIR="$DATA_DIR",IMAGE="SLTC002R_F" slurm/inference/inference_unet.slurm; sleep 1
sbatch --export=DATA_DIR="$DATA_DIR",IMAGE="SLTC004L_F" slurm/inference/inference_unet.slurm; sleep 1
sbatch --export=DATA_DIR="$DATA_DIR",IMAGE="SLTC004R_F" slurm/inference/inference_unet.slurm; sleep 1
sbatch --export=DATA_DIR="$DATA_DIR",IMAGE="SLTC005L_F" slurm/inference/inference_unet.slurm; sleep 1
sbatch --export=DATA_DIR="$DATA_DIR",IMAGE="SLTC005R_F" slurm/inference/inference_unet.slurm; sleep 1
sbatch --export=DATA_DIR="$DATA_DIR",IMAGE="SLTC006L_F" slurm/inference/inference_unet.slurm; sleep 1
sbatch --export=DATA_DIR="$DATA_DIR",IMAGE="SLTC006R_F" slurm/inference/inference_unet.slurm; sleep 1
sbatch --export=DATA_DIR="$DATA_DIR",IMAGE="SLTC007L_F" slurm/inference/inference_unet.slurm; sleep 1
sbatch --export=DATA_DIR="$DATA_DIR",IMAGE="SLTC007R_F" slurm/inference/inference_unet.slurm; sleep 1
sbatch --export=DATA_DIR="$DATA_DIR",IMAGE="SLTC008L_F" slurm/inference/inference_unet.slurm; sleep 1
sbatch --export=DATA_DIR="$DATA_DIR",IMAGE="SLTC008R_F" slurm/inference/inference_unet.slurm; sleep 1
sbatch --export=DATA_DIR="$DATA_DIR",IMAGE="SLTC010L_F" slurm/inference/inference_unet.slurm; sleep 1
sbatch --export=DATA_DIR="$DATA_DIR",IMAGE="SLTC010R_F" slurm/inference/inference_unet.slurm; sleep 1
sbatch --export=DATA_DIR="$DATA_DIR",IMAGE="SLTC013L_F" slurm/inference/inference_unet.slurm; sleep 1
sbatch --export=DATA_DIR="$DATA_DIR",IMAGE="SLTC013R_F" slurm/inference/inference_unet.slurm; sleep 1
sbatch --export=DATA_DIR="$DATA_DIR",IMAGE="SLTC014L_F" slurm/inference/inference_unet.slurm; sleep 1
sbatch --export=DATA_DIR="$DATA_DIR",IMAGE="SLTC014R_F" slurm/inference/inference_unet.slurm; sleep 1
sbatch --export=DATA_DIR="$DATA_DIR",IMAGE="SLTC015L_F" slurm/inference/inference_unet.slurm; sleep 1
sbatch --export=DATA_DIR="$DATA_DIR",IMAGE="SLTC015R_F" slurm/inference/inference_unet.slurm; sleep 1
sbatch --export=DATA_DIR="$DATA_DIR",IMAGE="SLTC018L_F" slurm/inference/inference_unet.slurm; sleep 1
sbatch --export=DATA_DIR="$DATA_DIR",IMAGE="SLTC018R_F" slurm/inference/inference_unet.slurm; sleep 1
sbatch --export=DATA_DIR="$DATA_DIR",IMAGE="SLTC019L_F" slurm/inference/inference_unet.slurm; sleep 1
sbatch --export=DATA_DIR="$DATA_DIR",IMAGE="SLTC019R_F" slurm/inference/inference_unet.slurm; sleep 1
sbatch --export=DATA_DIR="$DATA_DIR",IMAGE="SLTC020L_F" slurm/inference/inference_unet.slurm; sleep 1
sbatch --export=DATA_DIR="$DATA_DIR",IMAGE="SLTC020R_F" slurm/inference/inference_unet.slurm; sleep 1
sbatch --export=DATA_DIR="$DATA_DIR",IMAGE="SLTC021L_F" slurm/inference/inference_unet.slurm; sleep 1
sbatch --export=DATA_DIR="$DATA_DIR",IMAGE="SLTC021R_F" slurm/inference/inference_unet.slurm; sleep 1
sbatch --export=DATA_DIR="$DATA_DIR",IMAGE="SLTC023L_F" slurm/inference/inference_unet.slurm; sleep 1
sbatch --export=DATA_DIR="$DATA_DIR",IMAGE="SLTC023R_F" slurm/inference/inference_unet.slurm; sleep 1
sbatch --export=DATA_DIR="$DATA_DIR",IMAGE="SLTC025L_F" slurm/inference/inference_unet.slurm; sleep 1
sbatch --export=DATA_DIR="$DATA_DIR",IMAGE="SLTC025R_F" slurm/inference/inference_unet.slurm; sleep 1
sbatch --export=DATA_DIR="$DATA_DIR",IMAGE="SLTC027L_F" slurm/inference/inference_unet.slurm; sleep 1
sbatch --export=DATA_DIR="$DATA_DIR",IMAGE="SLTC027R_F" slurm/inference/inference_unet.slurm; sleep 1
sbatch --export=DATA_DIR="$DATA_DIR",IMAGE="SLTC030L_F" slurm/inference/inference_unet.slurm; sleep 1
sbatch --export=DATA_DIR="$DATA_DIR",IMAGE="SLTC030R_F" slurm/inference/inference_unet.slurm; sleep 1
sbatch --export=DATA_DIR="$DATA_DIR",IMAGE="SLTC032L_F" slurm/inference/inference_unet.slurm; sleep 1
sbatch --export=DATA_DIR="$DATA_DIR",IMAGE="SLTC032R_F" slurm/inference/inference_unet.slurm; sleep 1
sbatch --export=DATA_DIR="$DATA_DIR",IMAGE="SLTC033L_F" slurm/inference/inference_unet.slurm; sleep 1
sbatch --export=DATA_DIR="$DATA_DIR",IMAGE="SLTC033R_F" slurm/inference/inference_unet.slurm; sleep 1
sbatch --export=DATA_DIR="$DATA_DIR",IMAGE="SLTC034L_F" slurm/inference/inference_unet.slurm; sleep 1
sbatch --export=DATA_DIR="$DATA_DIR",IMAGE="SLTC034R_F" slurm/inference/inference_unet.slurm; sleep 1
sbatch --export=DATA_DIR="$DATA_DIR",IMAGE="SLTC036L_F" slurm/inference/inference_unet.slurm; sleep 1
sbatch --export=DATA_DIR="$DATA_DIR",IMAGE="SLTC036R_F" slurm/inference/inference_unet.slurm; sleep 1
sbatch --export=DATA_DIR="$DATA_DIR",IMAGE="SLTC037L_F" slurm/inference/inference_unet.slurm; sleep 1
sbatch --export=DATA_DIR="$DATA_DIR",IMAGE="SLTC037R_F" slurm/inference/inference_unet.slurm; sleep 1
sbatch --export=DATA_DIR="$DATA_DIR",IMAGE="SLTC038L_F" slurm/inference/inference_unet.slurm; sleep 1
sbatch --export=DATA_DIR="$DATA_DIR",IMAGE="SLTC038R_F" slurm/inference/inference_unet.slurm; sleep 1
sbatch --export=DATA_DIR="$DATA_DIR",IMAGE="SLTC039L_F" slurm/inference/inference_unet.slurm; sleep 1
sbatch --export=DATA_DIR="$DATA_DIR",IMAGE="SLTC039R_F" slurm/inference/inference_unet.slurm; sleep 1
sbatch --export=DATA_DIR="$DATA_DIR",IMAGE="SLTC040L_F" slurm/inference/inference_unet.slurm; sleep 1
sbatch --export=DATA_DIR="$DATA_DIR",IMAGE="SLTC040R_F" slurm/inference/inference_unet.slurm; sleep 1
sbatch --export=DATA_DIR="$DATA_DIR",IMAGE="SLTC041L_F" slurm/inference/inference_unet.slurm; sleep 1
sbatch --export=DATA_DIR="$DATA_DIR",IMAGE="SLTC041R_F" slurm/inference/inference_unet.slurm; sleep 1
sbatch --export=DATA_DIR="$DATA_DIR",IMAGE="SLTC045L_F" slurm/inference/inference_unet.slurm; sleep 1
sbatch --export=DATA_DIR="$DATA_DIR",IMAGE="SLTC045R_F" slurm/inference/inference_unet.slurm; sleep 1
sbatch --export=DATA_DIR="$DATA_DIR",IMAGE="SLTC047L_F" slurm/inference/inference_unet.slurm; sleep 1
sbatch --export=DATA_DIR="$DATA_DIR",IMAGE="SLTC047R_F" slurm/inference/inference_unet.slurm; sleep 1
sbatch --export=DATA_DIR="$DATA_DIR",IMAGE="SLTC049L_F" slurm/inference/inference_unet.slurm; sleep 1
sbatch --export=DATA_DIR="$DATA_DIR",IMAGE="SLTC049R_F" slurm/inference/inference_unet.slurm; sleep 1