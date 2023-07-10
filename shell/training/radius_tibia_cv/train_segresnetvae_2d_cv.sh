#!/bin/bash
for dropout in "0.1" "0.3" "0.5"
do
  for learning_rate in "0.001" "0.0001"
  do
    sbatch \
    --export=DROPOUT="$dropout",LEARNING_RATE="$learning_rate" \
    slurm/training/radius_tibia_cv/train_segresnetvae_2d_cv.slurm
    sleep 1
  done
done