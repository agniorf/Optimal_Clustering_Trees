#!/bin/bash
#SBATCH --array=1-24
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=32000
#SBATCH -p sched_mit_sloan_batch
#SBATCH --time=4-00:00
#
srun /home/software/julia/0.6.0/bin/julia ../src/run-main.jl $SLURM_ARRAY_TASK_ID
