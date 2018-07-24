using DataFrames, MLDataUtils
using Clustering, Distances
using RDatasets
using OptimalTrees
# using Gadfly
reload("OptimalTrees")

# include("evaluation_tools.jl")
# ls_data = readtable("../data/localSearch_2.csv");X = ls_data[1:2];  y = ones(size(X,1)); truelabels = false;

# plot(X, x = :V1, y = :V2)


# s = 2;
# cp = 0.12084827390805059;

# println("Running for cp = $(cp): ")
# lnr= OptimalTrees.OptimalTreeClassifier(ls_num_tree_restarts = 100, ls_random_seed = s,
# 	max_depth = 2, minbucket = 5, criterion = :density, show_progress_bar=true);
# OptimalTrees.fit!(lnr, X, y);
# OptimalTrees.showinbrowser(lnr)



# for cp in [0,.01,.05,.1]
# 	println("Running for cp = $(cp): ")
# 	lnr= OptimalTrees.OptimalTreeClassifier(ls_num_tree_restarts = 10, ls_random_seed = s,
# 		max_depth = 2, minbucket = 5, criterion = :density, show_progress_bar=true);
# 	OptimalTrees.fit!(lnr, X, y);
# 	OptimalTrees.showinbrowser(lnr)
# end

# ls_data = readtable("../data/localSearch_2.csv");X = ls_data[1:2];  y = ones(size(X,1)); truelabels = false;

atom = readtable("../data/Atom.csv"); X = atom[1:end-1]; y = atom[end];
# rusp = dataset("cluster", "ruspini"); X = rusp[1:end]; y = ones(size(rusp,1));


s = 2;
cr = :silhouette; 
lnr_greedy = OptimalTrees.OptimalTreeClassifier(localsearch = false, cp = 0.0,
	max_depth = 3, minbucket = 1, criterion = :dunnindex, show_progress_bar=true, ls_warmstart_criterion = :dunnindex);
OptimalTrees.fit!(lnr_greedy, X, y);
a = OptimalTrees.score(lnr_greedy, X, y);
OptimalTrees.showinbrowser(lnr_greedy)

lnr_local = OptimalTrees.OptimalTreeClassifier(ls_num_tree_restarts = 5, cp = 0.0, ls_random_seed = s,
	max_depth = 2, minbucket = 1, criterion = :dunnindex, show_progress_bar=true, ls_warmstart_criterion=:dunnindex);
OptimalTrees.fit!(lnr_local, X, y);
OptimalTrees.showinbrowser(lnr_local)

datafolderpath = "/Users/agni/Packages/Optimal_Clustering_Trees/data"
filepath = joinpath(datafolderpath, "localSearch_7.csv")
# reload("OptimalTrees");

data = readtable(filepath, makefactors = true);
X = data[:,1:2]
y = data[:,3]

lnr_local = OptimalTrees.OptimalTreeClassifier(ls_num_tree_restarts = 5, cp = 0.0, ls_random_seed = s,
	max_depth = 3, minbucket = 1, criterion = :silhouette, show_progress_bar=true, ls_warmstart_criterion=:silhouette);

lnr_local = OptimalTrees.OptimalTreeClassifier(ls_num_tree_restarts = 5, cp = 0.0, ls_random_seed = s,
	max_depth = 3, minbucket = 1, criterion = :dunnindex, show_progress_bar=true, ls_warmstart_criterion=:dunnindex);
OptimalTrees.fit!(lnr_local, X, y);
# OptimalTrees.showinbrowser(lnr_local)

lnr_local = OptimalTrees.OptimalTreeClassifier(ls_num_tree_restarts = 5, cp = 0.0, ls_random_seed = s,
	max_depth = 3, minbucket = 1, criterion = :silhouette, show_progress_bar=true, ls_warmstart_criterion=:silhouette);

lnr_local = OptimalTrees.OptimalTreeClassifier(ls_num_tree_restarts = 5, cp = 0.0, ls_random_seed = s,
	max_depth = 3, minbucket = 1, criterion = :dunnindex, show_progress_bar=true, ls_warmstart_criterion=:dunnindex);

OptimalTrees.fit!(lnr_local, X, y);

#Scoring functions
dunn_score(lnr_local, OptimalTrees.apply(lnr_local, X))
silhouette_score(lnr_local, OptimalTrees.apply(lnr_local, X))
OptimalTrees.score(lnr_local,X,y)
lnr_local.tree_.nodes[1].raw_error


rusp = dataset("cluster", "ruspini");
X = rusp[1:2]; y = ones(size(X,1));
lnr_local = OptimalTrees.OptimalTreeClassifier(ls_num_tree_restarts = 20, cp = 0.0, ls_random_seed = s,
	max_depth = 3, minbucket = 1, criterion = :dunnindex, show_progress_bar=true, ls_warmstart_criterion=:dunnindex);

OptimalTrees.fit!(lnr_local, X, y);
b = OptimalTrees.score(lnr_local, X, y);
OptimalTrees.showinbrowser(lnr_local)

#Scoring functions
dunn_score(X, OptimalTrees.apply(lnr_local, X))
OptimalTrees.score(lnr_local,X,y)




dunn_score(X, OptimalTrees.apply(lnr_local, X))

lnr_greedy = OptimalTrees.OptimalTreeClassifier(localsearch = false, cp = 0.0,
	max_depth = 3, minbucket = 1, criterion = :dunnindex, ls_warmstart_criterion=:dunnindex, show_progress_bar=true);

lnr_greedy = OptimalTrees.OptimalTreeClassifier(localsearch = false, cp = 0.0,
	max_depth = 3, minbucket = 1, criterion = :dunnindex, show_progress_bar=true);

OptimalTrees.fit!(lnr_greedy, X, y);
OptimalTrees.showinbrowser(lnr_greedy)

