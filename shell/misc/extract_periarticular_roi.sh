#!/bin/bash
for visit_num in 1 2 3 4
do
  for bone in "femur" # "tibia"
  do
    sbatch --export=VISIT_NUM="$visit_num",BONE="$bone" slurm/misc/extract_periarticular_roi.slurm
    sleep 1
  done
done