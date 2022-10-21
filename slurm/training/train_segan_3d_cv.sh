#!/bin/bash
for channels in "32 64 128" "64 128 256" "32 64 128 256" "64 128 256 512"
do
  for dropout in "0.1" "0.3" "0.5"
  do
    for learning_rate in "0.001" "0.01" "0.1"
    do
      sbatch \
      --export=CHANNELS="$channels",DROPOUT="$dropout",LEARNING_RATE="$learning_rate" \
      slurm/training/train_segan_3d_cv.slurm
      sleep 1
    done
  done
done