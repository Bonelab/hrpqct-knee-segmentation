#!/bin/bash
for dropout in "0.1" "0.3" "0.5"
do
  for learning_rate in "0.001" "0.0001"
  do
    sbatch \
    --export=DROPOUT="$dropout",LEARNING_RATE="$learning_rate" \
    slurm/training/train_segresnetvae_3d_cv.slurm
    sleep 1
  done
done