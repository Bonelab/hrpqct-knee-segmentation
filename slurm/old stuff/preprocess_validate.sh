#!/bin/bash
sbatch --export=IS=0,IE=100 slurm/preprocess_validate_base.slurm
sbatch --export=IS=100,IE=200 slurm/preprocess_validate_base.slurm
sbatch --export=IS=200,IE=300 slurm/preprocess_validate_base.slurm
sbatch --export=IS=300,IE=390 slurm/preprocess_validate_base.slurm
