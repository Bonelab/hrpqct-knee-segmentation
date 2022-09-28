#!/bin/bash
for i in {0..26}
do
  idx_start=$((100*i))
  idx_end=$((100*(i+1)))
  sbatch --export=IS="$idx_start",IE="$idx_end" slurm/preprocessing/preprocess_3d.slurm
  sleep 1
done