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

include("evaluation_pipeline_kmeans0.jl")
include("evaluation_pipeline_Rcompare.jl")
include("algorithm_comparison.jl")
include("../../evaluation_tools_full.jl")


method = "ICOT_local"; geom_search = true; threshold = .99; seed = 2; 
gridsearch = false; num_tree_restarts = 100; complexity = 0.0; min_bucket = 1; maxdepth = 3; 
datafolderpath = "../data/"; warm_start = :none; resultsfolderpath = "../results_R/"
normalize_R = true
  ###### STEP 1: READ THE DATA

name_short = "FHS"
data = CSV.read("../../../../../Dropbox (MIT)/research/clustering/Experiments/FHS/data/data_4_one_hot.csv");
X = data[1:(end-1)]; 
y = ifelse.(isodd.(collect(1:nrow(X))),1,2) ## dummy y to split two clusters so that score is a float; 

##### STEP 2: Compare to ground truth
true_assignments = Array{Int64}(y);
true_k = length(unique(y));
true_sil, true_dunn = test_performance_general(X, true_assignments);

results = DataFrame(seed = seed, data = name_short, criterion = cr,  
  geom_threshold = threshold, warm_start = warm_start,
  method = "True", K = Float64(true_k), silhouette = true_sil, dunn = true_dunn, 
  ari_true = 1.0, runtime = 0.0);


  ####### STEP 6: Comparison to alternative clustering methods
println("\nRunning Comparison Methods (cross-validate K)")
# true_k = length(unique(y)); 
# min_k = max(min(leaf_cnt,true_k)-2,2); max_k = max(true_k, leaf_cnt)+2;
min_k = 2; max_k = 10;
eps_range = .1:.1:5.0;

# Return a dictionary of scores and assignments for each k, as well as the best k value (max score)
for cr in [:silhouette, :dunnindex], m in ["kmeans_plus","hclust","gmm","dbscan"]
  println("Method = $m")
  k_range = ifelse(m == "dbscan", eps_range, min_k:max_k)
  run_time = @elapsed best_k, assignments = eval_method(X, k_range, seed, cr, m, normalize = normalize_R);
  sil, dunn = test_performance_general(X, assignments);

  ari_true = ifelse(true_k == best_k, randindex(true_assignments, assignments)[1], -10)

  ### Add results to master DF
  to_add = DataFrame(seed = seed, data = name_short, criterion = cr, 
    geom_threshold = threshold, warm_start = warm_start,
    method = m, K = best_k, silhouette = sil, dunn = dunn, 
    ari_true = ari_true, runtime = run_time)
  append!(results, to_add)
end

println(results)
#Results: Save results in an array to paste into Excel file 
filepath = "$name_short"
# filepath_lnr = joinpath(resultsfolderpath, "lnr-$dataset_name-$cr-$method-lnr.jld")
#    @save filepath_lnr lnr

CSV.write("$(resultsfolderpath)$filepath.csv", results)
