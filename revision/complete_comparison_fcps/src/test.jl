using CSV

println("Argument: " ARGS[1])

df = CSV.read("data_banknote_authentication.txt",
              header=[:variance, :skewness, :curtosis, :entropy, :class])

X = df[:, 1:4]
y = df[:, 5]
(train_X, train_y), (test_X, test_y) = IAI.split_data(:classification, X, y,
                                                      seed=1)

grid = IAI.GridSearch(
    IAI.OptimalTreeClassifier(
        random_seed=1,
    ),
    max_depth=1:5,
)
IAI.fit!(grid, train_X, train_y)
a = IAI.predict(grid, test_X)
println("Done with experiment

#!/bin/bash
#SBATCH --array=1-20
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4000
#SBATCH -p sched_mit_sloan_batch
#SBATCH --time=0-01:00
#
srun julia -J ~/software/julia-1.1.0/lib/julia/sys.so test_iai.jl $SLURMD_NODENAME
