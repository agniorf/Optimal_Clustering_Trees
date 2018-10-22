#!/bin/bash
#SBATCH --array=1-63
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=32000
#SBATCH -p sched_mit_sloan_batch
#SBATCH --time=2-00:00
#
srun /home/software/julia/0.6.0/bin/julia ../src/run-main.jl $SLURM_ARRAY_TASK_ID
