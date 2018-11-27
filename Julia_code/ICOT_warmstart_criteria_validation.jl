using DataFrames, MLDataUtils
using Clustering, Distances
using RDatasets
using ICOT
using OptimalTrees
# using Gadfly
using BenchmarkTools


# dataset = readtable("../Experiments/Experiment_1/data/Lsun.csv"); 
# dataset = readtable("../Experiments/Experiment_1/data/Lsun.csv"); 
# dataset = readtable("../Testing_Class_Project/data/TwoDiamonds.csv")
# dataset = readtable("../data/localSearch_6.csv");
# X = dataset[1:end-1]; y = dataset[end];

# data = dataset("cluster", "ruspini");
data = readtable("../data/fscs/Lsun.csv"); 
K = 3;

s = 2;
cr = :dunnindex; 
bucket = 1; depth = 3; restarts = 10;

data_array = convert(Array{Float64}, data);
n, p = size(data_array)
data_t = data_array';

srand(s);
kmeans_result = kmeans(data_t, K);

assignments = kmeans_result.assignments;
data_full = DataFrame(hcat(data, assignments));
rename!(data_full, :x1, :kmean_assign);
# plot(dataset_full, x = :V2, y = :V3, color = :kmean_assign)

X = data_full[1:p]; y = data_full[end];


## RUN optimal trees independently
# lnr_oct = OptimalTrees.OptimalTreeClassifier(localsearch = false,
# 	max_depth = 3, minbucket = 1, criterion = :misclassification);
# @time OptimalTrees.fit!(lnr_oct, X, y);
# OptimalTrees.showinbrowser(lnr_oct);

# reload("OptimalTrees")
lnr_greedy = ICOT.OptimalTreeClassifier(localsearch = false, cp = 0.0,
	max_depth = depth, minbucket = bucket, criterion = cr, ls_warmstart_criterion = cr,
	kmeans_warmstart = false);
@time ICOT.fit!(lnr_greedy, X, y);
score_greedy = ICOT.score(lnr_greedy, X, y);

ICOT.showinbrowser(lnr_greedy)


# lnr_local = ICOT.OptimalTreeClassifier(ls_num_tree_restarts = restarts, cp = 0.0, ls_random_seed = s,
# 	max_depth = depth, minbucket = bucket, criterion = cr, show_progress_bar=true, 
# 	ls_warmstart_criterion= cr, geom_search=false, geom_threshold=0.0, 
# 	kmeans_warmstart = false);
# @time ICOT.fit!(lnr_local, X, y);
# score_local = ICOT.score(lnr_local, X, y);

# ICOT.showinbrowser(lnr_local)

lnr_local = ICOT.OptimalTreeClassifier(ls_num_tree_restarts = restarts, cp = 0.0, ls_random_seed = s,
	max_depth = depth, minbucket = bucket, criterion = cr, show_progress_bar=true, 
	ls_warmstart_criterion= cr, geom_search=true, geom_threshold=0.9, 
	kmeans_warmstart = false);
@btime ICOT.fit!(lnr_local, X, y);
score_local = ICOT.score(lnr_local, X, y);

ICOT.showinbrowser(lnr_local)

cr = :dunnindex; 

lnr_ws = ICOT.OptimalTreeClassifier(ls_num_tree_restarts = restarts, cp = 0.0, ls_random_seed = s,
	max_depth = depth, minbucket = bucket, criterion = cr, show_progress_bar=true, 
	ls_warmstart_criterion= cr, geom_search=true, geom_threshold=0.9,
	kmeans_warmstart = true);
ICOT.fit!(lnr_ws, X, y);
score_ws = ICOT.score(lnr_ws, X, y);
score_al_ws = ICOT.score(lnr_ws, X, y, criterion=:silhouette);

lnr_ws.tree_.dunnindex_score
lnr_ws.tree_.silhouette_score

ICOT.apply(lnr_ws, X, lnr_ws.tree_)

ICOT.showinbrowser(lnr_ws)

