include("../src/evaluation_tools.jl")
include("../src/evaluation_pipeline_exp.jl")

datasets = readdir("../data/random/")
criter=[:silhouette,:dunnindex,:robustdunn]
clust_method=["localsearch","greedy"]

datalistcriterionpairs =collect(Iterators.product(datasets,criter,clust_method))

function main(;seed::Int64=2,
               gridsearch::Bool=false,
               num_tree_restarts::Int64=100,
               complexity::Float64= 0.0,
               minbucket::Int64=1,
               datafolderpath::String="../data/random/",
               resultsfolderpath::String="../results/")
  arg_in = parse(Int64, ARGS[1])
  (data, criterion, method) = datalistcriterionpairs[arg_in]
  run_single(;data=data,
             cr=criterion,
             method = method,
             seed=2,
             gridsearch=false,
             num_tree_restarts=100,
             complexity= 0.0,
             min_bucket=minbucket,
             maxdepth=6,
             datafolderpath=datafolderpath,
             resultsfolderpath=resultsfolderpath)
end
