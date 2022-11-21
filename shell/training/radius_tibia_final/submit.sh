#!/bin/bash
sbatch slurm/training/radius_tibia_final/train_segan_2d_final.slurm
sleep 1
sbatch slurm/training/radius_tibia_final/train_segan_3d_final.slurm
sleep 1