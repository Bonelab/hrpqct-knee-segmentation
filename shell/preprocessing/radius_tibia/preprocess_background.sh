#!/bin/bash
study="NORMXTII"
site="radius"
sbatch --export=IS=0,IE=115,STUDY="$study",SITE="$site" slurm/preprocessing/preprocess_3d_background.slurm
sleep 1
site="tibia"
sbatch --export=IS=0,IE=115,STUDY="$study",SITE="$site" slurm/preprocessing/preprocess_3d_background.slurm
sleep 1
