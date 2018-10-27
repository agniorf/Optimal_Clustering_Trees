using DataFrames, MLDataUtils
using Clustering, Distances
using RDatasets
using OptimalTrees
using Gadfly
using BenchmarkTools

dataset = readtable("data.csv"); 
seed=1;

X = dataset; 
y = ones(size(dataset,1)); 
truelabels = false; 
srand(seed);

col = [:gender, :diabetes]
pool!(X,col)

method ="localsearch" 
num_tree_restarts = 1
complexity = 0.0
maxdepth = 3
min_bucket = 5
cr = :silhouette
geom_yn = true
gt = 0.9


lnr = OptimalTrees.OptimalTreeClassifier(ls_num_tree_restarts = num_tree_restarts, ls_random_seed = seed, cp = complexity, max_depth = maxdepth,
		minbucket = min_bucket, criterion = cr, ls_warmstart_criterion = cr, 
		geom_search = geom_yn, geom_threshold = gt, a = a);
OptimalTrees.fit!(lnr, X, y);
OptimalTrees.showinbrowser(lnr)
