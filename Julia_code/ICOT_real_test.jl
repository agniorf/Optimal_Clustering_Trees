using DataFrames, MLDataUtils
using Clustering, Distances
using RDatasets
using OptimalTrees
# using Gadfly
# using BenchmarkTools


# dataset = readtable("../Experiments/Experiment_1/data/Lsun.csv"); 
# dataset = readtable("../Experiments/Experiment_1/data/Lsun.csv"); 
# dataset = readtable("../Testing_Class_Project/data/TwoDiamonds.csv")
# dataset = readtable("../data/localSearch_6.csv");
# X = dataset[1:end-1]; y = dataset[end];

data = readtable("random_hubway_sample_seed1_obs_50000.csv")
X = data[1:end]; y = ones(size(data,1));


s = 2;
cr = :silhouette; 

# reload("OptimalTrees")
lnr_greedy = OptimalTrees.OptimalTreeClassifier(localsearch = false, cp = 0.0,
	max_depth = 3, minbucket = 1, criterion = cr, show_progress_bar=true, ls_warmstart_criterion = cr);
@time OptimalTrees.fit!(lnr_greedy, X, y);
a = OptimalTrees.score(lnr_greedy, X, y);


# OptimalTrees.showinbrowser(lnr_greedy)
reload("OptimalTrees")
lnr_local = OptimalTrees.OptimalTreeClassifier(ls_num_tree_restarts = 10, cp = 0.0, ls_random_seed = s,
	max_depth = 6, minbucket = 10, criterion = cr, show_progress_bar=true, ls_warmstart_criterion= cr, geom_search=true, geom_threshold=0.9);
@time OptimalTrees.fit!(lnr_local, X, y);
b = OptimalTrees.score(lnr_local, X, y);

OptimalTrees.showinbrowser(lnr_local)
