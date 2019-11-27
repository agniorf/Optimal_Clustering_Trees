#!/bin/bash
#SBATCH --array=1-90
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=16000
#SBATCH -p sched_mit_sloan_batch
#SBATCH --time=2-00:00
#
export IAI_LICENSE_FILE="$HOME/iai-licenses/${SLURMD_NODENAME}.lic"
srun julia -J ~/software/julia-1.1.0/lib/julia/sys.so ../src/run-main.jl $SLURM_ARRAY_TASK_ID
