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

include("../../../ICOT/src/ICOT.jl")
include("../evaluation_tools_full.jl")

datasets = ["Atom", "Chainlink", "EngyTime",
 "Hepta", "Lsun", "Target",
 "Tetra", "TwoDiamonds", "WingNut"];
criter=[:silhouette,:dunnindex];

results = DataFrame(data = String[], criterion = Symbol[], 
  method = String[], K = Int64[], score = Float64[], runtime = Float64[]);

for data_name in datasets, cr in criter
	run_time = @elapsed best_k, score, assignments = eval_clus(data_name, cr);
	to_add = DataFrame(data = data_name, criterion = cr, 
	  method = "Clus", K = best_k, score = score, runtime = run_time);
	append!(results, to_add)
end


function parse_clus(data_name, depth)
	## Generally > 10 sufficient for km_it_cnt, em_it_cnt
	dims = size(CSV.read("data/$(data_name).csv"))
	p = dims[2]-1
	@rput data_name
	@rput depth
	@rput p
	R"
	library(foreign)
	library(tidyverse)
	result_file <- read.arff(paste0(data_name,\"_depth\",depth,\".train.1.pred.arff\"))
	X <- result_file[,1:p]
	assignments <- result_file$`Pruned-models` %>% as.integer()
	"
	@rget X
	@rget assignments
	return X, Array{Int64}(assignments)
end


function eval_clus(data_name, cr; param_range = 1:3, normalize = false)
	score_dict = Dict{Float64,Float64}()
	assignments_dict = Dict{Float64,Array{Int64}}()
	for depth in param_range
		X, assignments = parse_clus(data_name, depth);
		distance_matrix = create_distance_matrix_numeric(convert(Matrix{Float64},X));
		score_dict[depth] = cluster_score(distance_matrix, assignments, cr)
		assignments_dict[depth] = assignments
	end
	bestk = collect(keys(score_dict))[findmax(collect(values(score_dict)))[2]]
	println("Best K = $bestk: Score = $(round(score_dict[bestk],digits=3))")
	return bestk, score_dict[bestk], assignments_dict[bestk]
end