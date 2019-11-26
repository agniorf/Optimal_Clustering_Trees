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

datasets = ["Atom.csv", "Chainlink.csv", "EngyTime.csv",
 "Hepta.csv", "Lsun.csv", "Target.csv",
 "Tetra.csv", "TwoDiamonds.csv", "WingNut.csv"];

criter=[:silhouette,:dunnindex]

#Optimized or not?
clust_method=["ICOT_local"]

#K-means warm start
warm_start = [:oct]

#Geomsearch or Not
geom_search = [true]

#Geometric thresholds
thresholds = [.99]

#
# seedSplitList = [2]
seedSplitList = [1,3,4,5]

paramList = collect(Iterators.product(datasets, criter, clust_method, warm_start, geom_search, thresholds, seedSplitList))[:]

arg_in = parse(Int64, ARGS[1])
(data, criterion, clust_method, warm_start, geom_search, threshold, seed) = paramList[arg_in]


# dataset_name = deepcopy(data);
# name_short = split(dataset_name, ".")[1];
# data_path = joinpath(datafolderpath, data)

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
 
