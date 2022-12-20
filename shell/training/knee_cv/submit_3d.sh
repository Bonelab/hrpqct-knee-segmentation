#!/bin/bash
sbatch slurm/training/knee_cv/train_unet_3d_cv.slurm
sleep 1
sbatch slurm/training/knee_cv/train_unetpp_3d_cv.slurm
sleep 1
sbatch slurm/training/knee_cv/train_unetr_3d_cv.slurm
sleep 1
sbatch slurm/training/knee_cv/train_segan_3d_cv.slurm
sleep 1
sbatch slurm/training/knee_cv/train_segresnetvae_3d_cv.slurm
sleep 1