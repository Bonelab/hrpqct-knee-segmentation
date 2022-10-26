#!/bin/bash
for channels in "16 32 64 128" "32 64 128 256" "64 128 256 512"
do
  for dropout in "0.1" "0.3" "0.5"
  do
    for learning_rate in "0.001" "0.0001"
    do
      sbatch \
      --export=CHANNELS="$channels",DROPOUT="$dropout",LEARNING_RATE="$learning_rate" \
      slurm/training/train_unet_3d_cv.slurm
      sleep 1
    done
  done
done