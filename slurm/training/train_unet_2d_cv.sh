#!/bin/bash
channels="32 64 128 256"
dropout="0.1"
learning_rate="0.001"
sbatch \
--export=CHANNELS="$channels",DROPOUT="$dropout",LEARNING_RATE="$learning_rate" \
slurm/training/train_unet_2d_cv.slurm