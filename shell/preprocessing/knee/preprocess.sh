#!/bin/bash
study="SALTAC"
for visit_num in 1 2 3 4
do
  for bone in "femur" "tibia"
  do
    for compartment in "medial" "lateral"
    do
      sbatch --export=STUDY="$study",VISIT_NUM="$visit_num",BONE="$bone",COMPARTMENT="$compartment" slurm/preprocessing/knee/preprocess_3d.slurm
      sleep 1
    done
  done
done