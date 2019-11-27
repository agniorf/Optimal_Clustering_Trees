using DataFrames, MLDataUtils
using Clustering, Distances
using BenchmarkTools
using RCall
using Clustering
using Statistics
using CSV
using Random
using Logging

logger = Logging.SimpleLogger(stderr,Logging.Warn)
global_logger(logger)

include("../../../../ICOT/src/ICOT.jl")

include("evaluation_pipeline_full_compare.jl")
include("algorithm_comparison.jl")
include("../../evaluation_tools_full.jl")

seedList = 1:5

datasets = ["Atom.csv", "Chainlink.csv", "EngyTime.csv",
 "Hepta.csv", "Lsun.csv", "Target.csv",
 "Tetra.csv", "TwoDiamonds.csv", "WingNut.csv"];

criter=[:silhouette,:dunnindex]

results = DataFrame(seed = Int64[], data = String[], criterion = Symbol[],  
    method = String[], K = Float64[], silhouette = Float64[], dunn = Float64[], 
    ari_true = Float64[], runtime = Float64[]);

for seed in seedList, data in datasets, cr in criter
  ###### STEP 1: READ THE DATA
  dataset_name = deepcopy(data);
  name_short = split(dataset_name, ".")[1];

  data_path = joinpath(datafolderpath, data)
  data = CSV.read(data_path); 
  X = data[1:(end-1)]; 
  y = data[:label]; 
  truelabels = true; 

  ##### STEP 2: Compare to ground truth
  true_assignments = Array{Int64}(y);
  true_k = length(unique(y));

  println("\nRunning Comparison Methods (cross-validate K)")
  eps_range = .1:.1:1.0;

  # Return a dictionary of scores and assignments for each k, as well as the best k value (max score)
  m = "hclust"
  println("Method = $m")
  k_range = ifelse(m == "dbscan", eps_range, min_k:max_k)
  run_time = @elapsed best_k, assignments = eval_method(X, k_range, seed, cr, m);
  sil, dunn = test_performance_general(X, assignments);

  ari_true = ifelse(true_k == best_k, randindex(true_assignments, assignments)[1], -10)

  ### Add results to master DF
  to_add = DataFrame(seed = seed, data = name_short, criterion = cr, 
  method = m, K = best_k, silhouette = sil, dunn = dunn, 
  ari_true = ari_true, runtime = run_time);
  append!(results, to_add)
end

CSV.write("$(resultsfolderpath)all_hclust.csv", results)


