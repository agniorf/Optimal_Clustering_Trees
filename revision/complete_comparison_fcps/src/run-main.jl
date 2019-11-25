using DataFrames, MLDataUtils
using Clustering, Distances
using BenchmarkTools
using RCall
using CSV
using Random
using Logging
using Statistics

logger = Logging.SimpleLogger(stderr,Logging.Warn)
global_logger(logger)

# include("../../../../../../git/ICOT/src/ICOT.jl")
include("../../../../../home/hwiberg/research/ICOT/src/ICOT.jl")

include("../src/evaluation_tools_full.jl")
include("../src/evaluation_pipeline_full.jl")

#Read in the datasets
datasets = ["Atom.csv", "Chainlink.csv", "EngyTime.csv",
 "Hepta.csv", "Lsun.csv", "Target.csv",
 "Tetra.csv", "TwoDiamonds.csv", "WingNut.csv"];
# datasets = ["Lsun.csv"]

#Read in the criteria
criter=[:silhouette,:dunnindex]
# criter=[:dunnindex]

#Optimized or not?
# clust_method=["ICOT_local","ICOT_greedy"]
clust_method=["ICOT_local"]

#K-means warm start
warm_start = [:oct]
# warm_start = [true]

#Geomsearch or Not
# geom_search = [true,false]
geom_search = [true]

#Geometric thresholds
# thresholds = [0, .9,.99]
# thresholds = [.9,.99]
thresholds = [.99]

#seed
# seedSplitList = [1:5;]
seedSplitList = [2]

paramList = collect(Iterators.product(datasets, criter, clust_method, warm_start, geom_search, thresholds, seedSplitList))[:]

arg_in = parse(Int64, ARGS[1])
(data, criterion, clust_method, warm_start, geom_search, threshold, seed) = paramList[arg_in]

println("Experiment: ", arg_in)
println("Dataset: ", data)
println("Criterion: ", criterion)
# println("Local search Method: ", clust_method)
# println("Kmeans warm start: ", warm_start)
# println("Geometric search: ", geom_search)
# println("Threshold for geometric: ", threshold)
println("Seed: ", seed)

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
             resultsfolderpath="../results/")


