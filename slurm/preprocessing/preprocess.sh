#!/bin/bash
study="NORMXTII"
for site in radius tibia
do
  for i in {0..11}
  do
    idx_start=$((100*i))
    idx_end=$((100*(i+1)))
    #sbatch --export=IS="$idx_start",IE="$idx_end",STUDY="$study",SITE="$site" slurm/preprocessing/preprocess_2d.slurm
    #sleep 1
    sbatch --export=IS="$idx_start",IE="$idx_end",STUDY="$study",SITE="$site" slurm/preprocessing/preprocess_2-5d.slurm
    sleep 1
    #sbatch --export=IS="$idx_start",IE="$idx_end",STUDY="$study",SITE="$site" slurm/preprocessing/preprocess_3d.slurm
    #sleep 1
  done
done
study="HIPFX"
for site in radius tibia
do
  for i in {0..1}
  do
    idx_start=$((100*i))
    idx_end=$((100*(i+1)))
    #sbatch --export=IS="$idx_start",IE="$idx_end",STUDY="$study",SITE="$site" slurm/preprocessing/preprocess_2d.slurm
    #sleep 1
    sbatch --export=IS="$idx_start",IE="$idx_end",STUDY="$study",SITE="$site" slurm/preprocessing/preprocess_2-5d.slurm
    sleep 1
    #sbatch --export=IS="$idx_start",IE="$idx_end",STUDY="$study",SITE="$site" slurm/preprocessing/preprocess_3d.slurm
    #sleep 1
  done
done
