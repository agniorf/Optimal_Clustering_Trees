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
include("algorithm_comparison.jl")
include("../../evaluation_tools_full.jl")


datasets = ["Atom.csv", "Chainlink.csv", "EngyTime.csv",
 "Hepta.csv", "Lsun.csv", "Target.csv",
 "Tetra.csv", "TwoDiamonds.csv", "WingNut.csv"];
criter=[:silhouette,:dunnindex];
clust_method=["ICOT_local"];
# warm_start = [:none];
# geom_search = [true];
# thresholds = [0.0,0.9,0.99];
warm_start = [:oct];
geom_search = [true];
thresholds = [0.99];
seedSplitList = [1,2,3,4,5];

paramList = collect(Iterators.product(datasets, criter, clust_method, warm_start, geom_search, thresholds, seedSplitList))[:]

arg_in = parse(Int64, ARGS[1])
(data, criterion, clust_method, warm_start, geom_search, threshold, seed) = paramList[arg_in]

println("Experiment: ", arg_in)
println("Dataset: ", data)
println("Seed: ", seed)
println("Validation criterion: ", criterion)

run_single(;data=data,
             cr=criterion,
             method = clust_method,
             warm_start = warm_start,
             geom_search = geom_search,
             threshold = threshold,
             seed=seed,
             gridsearch=false,
             num_tree_restarts=100,
             complexity= 0.0,
             min_bucket=1,
             maxdepth=3,
             datafolderpath="../data/",
             resultsfolderpath="../results_kmeans0/")
 
