#!/bin/bash
for channels in "32 64 128 256" "64 128 256 512" "32 64 128 256 512"
do
  for dropout in "0.1" "0.3" "0.5"
  do
    sbatch \
    --export=CHANNELS="$channels",DROPOUT="$dropout" slurm/training/train_unet_2d_cv.slurm
    sleep 1
  done
done


