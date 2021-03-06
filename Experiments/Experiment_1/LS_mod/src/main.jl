include("../src/evaluation_tools.jl")
include("../src/evaluation_pipeline_exp.jl")

datasets = readdir("../data/")
criter=[:silhouette,:dunnindex,:robustdunn]
clust_method=["localsearch"]
thresholds = [.25,.5,.75,.9]

datalistcriterionpairs_withgeom =collect(Iterators.product(datasets,criter,clust_method,true,thresholds));
datalistcriterionpairs_nogeom =collect(Iterators.product(datasets,criter,clust_method,false,0.0));
datalistcriterionpairs = cat(4, datalistcriterionpairs_withgeom, datalistcriterionpairs_nogeom);

function main(;seed::Int64=2,
               gridsearch::Bool=false,
               num_tree_restarts::Int64=100,
               complexity::Float64= 0.0,
               minbucket::Int64=1,
               datafolderpath::String="../data/",
               resultsfolderpath::String="../results/")
  arg_in = parse(Int64, ARGS[1])
  (data, criterion, method, geom_bool, gt_threshold) = datalistcriterionpairs[arg_in]
  run_single(;data=data,
             cr=criterion,
             method = method,
             seed=2,
             gridsearch=false,
             num_tree_restarts=100,
             complexity= 0.0,
             min_bucket=1,
             maxdepth=3,
             geom_yn = geom_bool,
             gt = gt_threshold,
             datafolderpath=datafolderpath,
             resultsfolderpath=resultsfolderpath)
end
