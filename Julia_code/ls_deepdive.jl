using DataFrames, MLDataUtils
using Clustering, Distances
using RDatasets
using OptimalTrees
using Gadfly
# reload("OptimalTrees")

include("evaluation_tools.jl")
ls_data = readtable("../data/localSearch_2.csv");X = ls_data[1:2];  y = ones(size(X,1)); truelabels = false;

plot(X, x = :V1, y = :V2)


s = 2;
cp = 0.12084827390805059;

println("Running for cp = $(cp): ")
lnr= OptimalTrees.OptimalTreeClassifier(ls_num_tree_restarts = 100, ls_random_seed = s,
	max_depth = 2, minbucket = 5, criterion = :density, show_progress_bar=true);
OptimalTrees.fit!(lnr, X, y);
OptimalTrees.showinbrowser(lnr)



for cp in [0,.01,.05,.1]
	println("Running for cp = $(cp): ")
	lnr= OptimalTrees.OptimalTreeClassifier(ls_num_tree_restarts = 10, ls_random_seed = s,
		max_depth = 2, minbucket = 5, criterion = :density, show_progress_bar=true);
	OptimalTrees.fit!(lnr, X, y);
	OptimalTrees.showinbrowser(lnr)
end