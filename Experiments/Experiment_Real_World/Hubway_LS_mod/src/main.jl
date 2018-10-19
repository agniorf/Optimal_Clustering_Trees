include("../src/evaluation_tools.jl")
include("../src/evaluation_pipeline_exp.jl")

# datasets = readdir("../data/")
# criter=[:silhouette,:dunnindex,:robustdunn]
# clust_method=["localsearch"]
# thresholds = [.75,.9]


# # datalistcriterionpairs_withgeom =collect(Iterators.product(datasets,criter,clust_method,true,thresholds));
# # datalistcriterionpairs_nogeom =collect(Iterators.product(datasets,criter,clust_method,false,0.0));
# # datalistcriterionpairs = cat(4, datalistcriterionpairs_withgeom, datalistcriterionpairs_nogeom);

function main(;seed::Int64=2,
               gridsearch::Bool=false,
               num_tree_restarts::Int64=100,
               complexity::Float64= 0.0,
               minbucket::Int64=1,
               datafolderpath::String="../data/",
               resultsfolderpath::String="../results/")
  arg_in = parse(Int64, ARGS[1])
  # (data, criterion, method, geom_bool, gt_threshold) = datalistcriterionpairs[arg_in]
  data = "random_hubway_sample_seed1_obs_5000.csv"; criter = :silhouette; method = "localsearch";
  geom_bool = true;  theshold = .9; 
  run_single(;data=data,
             cr=criterion,
             method = method,
             seed=seed,
             gridsearch=gridsearch,
             num_tree_restarts=num_tree_restarts,
             complexity= complexity,
             min_bucket=minbucket,
             maxdepth=4,
             geom_yn = geom_bool,
             gt = gt_threshold,
             datafolderpath=datafolderpath,
             resultsfolderpath=resultsfolderpath)
end
