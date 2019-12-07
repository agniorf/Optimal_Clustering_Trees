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

failed_params = CSV.read("../results/failed_parameters_engytime.csv")

arg_in = parse(Int64, ARGS[1])

clust_method = "ICOT_local"; geom_search = true;

data = failed_params[:data][arg_in]*".csv"
criterion = Symbol(failed_params[:criterion][arg_in])
warm_start = Symbol(failed_params[:warm_start][arg_in])
threshold = failed_params[:geom_threshold][arg_in]
seed  = failed_params[:seed][arg_in]

println("Experiment: ", arg_in)
println("Dataset: ", data)
println("Seed: ", seed)
println("Validation criterion: ", criterion)
println("Warm Start: ", warm_start)
println("Threshold: ", threshold)

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
 
# cr=criterion; 
# method = clust_method;
# gridsearch=false;
# num_tree_restarts=100;
# complexity= 0.0;
# min_bucket=1;
# maxdepth=3;
# datafolderpath="../data/";
# resultsfolderpath="../results/"
