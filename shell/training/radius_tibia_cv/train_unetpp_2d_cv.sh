#!/bin/bash
for channels in "16 16 32 64 32 16" "32 32 64 128 64 32"
do
  for dropout in "0.1" "0.3" "0.5"
  do
    for learning_rate in "0.001" "0.0001"
    do
      sbatch \
      --export=CHANNELS="$channels",DROPOUT="$dropout",LEARNING_RATE="$learning_rate" \
      slurm/training/radius_tibia_cv/train_unetpp_2d_cv.slurm
      sleep 1
    done
  done
done