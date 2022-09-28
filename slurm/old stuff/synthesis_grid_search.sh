#! /bin/bash

SLURM_DIR="slurm"
BASE_SCRIPT="synthesis_grid_search_base.slurm"
JOB_DIR="synthesis_gs"

declare -a MODEL_FILTERS_LIST=("4 4" "4 4 4" "4 4 4 4" "8 8" "8 8 8" "8 8 8 8" "8 16 8" "16 16" "16 16 16" "16 32 16")
declare -a DROPOUT_LIST=("0.1" "0.2" "0.3" "0.4" "0.5")
declare -a CHANNELS_LIST=("2" "4" "8" "16")

mkdir ${SLURM_DIR}/${JOB_DIR}

for ((i = 0; i < ${#MODEL_FILTERS_LIST[@]}; i++)); do
  for ((j = 0; j < ${#DROPOUT_LIST[@]}; j++)); do
    for ((k = 0; k < ${#CHANNELS_LIST[@]}; k++)); do
      MODEL_FILTERS=${MODEL_FILTERS_LIST[$i]}
      DROPOUT=${DROPOUT_LIST[$j]}
      CHANNELS=${CHANNELS_LIST[$k]}
      JOB_SLURM="${SLURM_DIR}/${JOB_DIR}/JOB_${i}_${j}_${k}.slurm"
      cp ${SLURM_DIR}/${BASE_SCRIPT} ${JOB_SLURM}
      sed -i "s/FILTERS/${MODEL_FILTERS}/" ${JOB_SLURM}
      sed -i "s/CHANNELS/${CHANNELS}/" ${JOB_SLURM}
      sed -i "s/DROPOUT/${DROPOUT}/" ${JOB_SLURM}
      sbatch ${JOB_SLURM}
      sleep 5
    done
  done
done
