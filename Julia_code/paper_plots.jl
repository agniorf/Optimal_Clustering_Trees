 using RDatasets, MLDataUtils
using Clustering
using Distances
using Gadfly

X = dataset("cluster", "ruspini"); y = ones(size(X,1))

num_tree_restarts = 1; seed = 2; complexity = 0; maxdepth = 3; min_bucket = 1; cr = :silhouette;

lnr = OptimalTrees.OptimalTreeClassifier(ls_num_tree_restarts = num_tree_restarts, ls_random_seed = seed, 
	cp = complexity, max_depth = maxdepth, minbucket = min_bucket, criterion = cr, ls_warmstart_criterion = cr);

OptimalTrees.fit!(lnr, X, y);
OptimalTrees.showinbrowser(lnr);

plot(X, x = :X, y = :Y, color = [colorant"black"])