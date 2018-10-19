include("../src/main.jl")
main(;seed=2,
      gridsearch=false,
      num_tree_restarts=100,
      complexity= 0.0,
      minbucket=25,
      datafolderpath="../data/",
      resultsfolderpath="../results/newdunn_min25/")
