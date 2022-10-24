#!/bin/bash
for channels in "16 32 64 128" "32 64 128 256" "64 128 256 512" "64 128 256 512 1024"
do
  for dropout in "0.1" "0.3" "0.5"
  do
    sbatch \
    --export=CHANNELS="$channels",DROPOUT="$dropout" slurm/training/train_segan_3d_cv.slurm
    sleep 1
  done
done