#!/bin/bash
for channels in "32 64 128 256" "32 64 128 256 512" "64 128 256 512" "64 128 256 512 1024"
do
  for dropout in "0.1" "0.2" "0.3" "0.4" "0.5"
  do
    for learning_rate in "0.0001" "0.001" "0.01" "0.1"
    do
      sbatch \
      --export=CHANNELS="$channels",DROPOUT="$dropout",LEARNING_RATE="$learning_rate" \
      slurm/training/train_unet_2d_cv.slurm
    done
  done
done


