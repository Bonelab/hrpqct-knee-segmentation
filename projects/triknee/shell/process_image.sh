#!/bin/bash
IMAGE=$1
BONE=$2
LEFT=$3
jid=$(sbatch --export=IMAGE=${IMAGE},BONE=${BONE},LEFT=${LEFT} projects/triknee/slurm/0_preinference.slurm | tr -dc "0-9")
jid=$(sbatch --export=IMAGE=${IMAGE},BONE=${BONE},LEFT=${LEFT} --dependency=afterany:${jid} projects/triknee/slurm/1_inference.slurm | tr -dc "0-9")
sbatch --export=IMAGE=${IMAGE},BONE=${BONE},LEFT=${LEFT} --dependency=afterany:${jid} projects/triknee/slurm/2_postinference.slurm