#!/bin/bash
#SBATCH --array=1-14
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=16000
#SBATCH -p sched_mit_sloan_batch
#SBATCH --time=2-00:00
#
srun /home/software/julia/0.6.0/bin/julia ../src/run-main.jl $SLURM_ARRAY_TASK_ID
