#!/bin/bash
for channels in "32 64 128 256" "64 128 256 512" "32 64 128 256 512"
do
  for dropout in "0.1" "0.3" "0.5"
  do
    for learning_rate in "0.001" "0.0001"
    do
      sbatch \
      --export=CHANNELS="$channels",DROPOUT="$dropout",LEARNING_RATE="$learning_rate" \
      slurm/training/train_unet_2d_cv.slurm
      sleep 1
    done
  done
done


