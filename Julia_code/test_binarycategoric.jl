using DataFrames, MLDataUtils
using Clustering, Distances
using RDatasets
using OptimalTrees
using Gadfly
using BenchmarkTools

data_full = readtable("../data/hubway_2000_example.csv")
data_full[:Time] = ifelse.(data_full[:Morning] .== 1, "Morning",
	ifelse.(data_full[:Afternoon] .== 1, "Afternoon",
		ifelse.(data_full[:Evening] .== 1, "Evening",
			ifelse.(data_full[:Night] .== 1, "Night", "Other"))));
data_full[:DayType] = ifelse.(data_full[:Weekday] .== 1, "Weekday", "Weekend");

X = data_full[1:2]; y = data_full[end];

cols = [:Time, :DayType, :Male, :Age, :Duration];
X = data_full[cols];
pool!(X, [:Time, :DayType])


s = 2;
cr = :silhouette; 

# reload("OptimalTrees")
lnr_greedy = OptimalTrees.OptimalTreeClassifier(localsearch = false, cp = 0.0,
	max_depth = 3, minbucket = 1, criterion = cr, show_progress_bar=true, ls_warmstart_criterion = cr);
@btime OptimalTrees.fit!(lnr_greedy, X, y);
a = OptimalTrees.score(lnr_greedy, X, y);


# OptimalTrees.showinbrowser(lnr_greedy)
reload("OptimalTrees")
lnr_local = OptimalTrees.OptimalTreeClassifier(ls_num_tree_restarts = 10, cp = 0.0, ls_random_seed = s,
	max_depth = 3, minbucket = 2, criterion = cr, show_progress_bar=true, ls_warmstart_criterion= cr, geom_search=true, geom_threshold=0.9);
>>>>>>> 27872488e259671ad1f781cf525b8b7cea3a3e08
@btime OptimalTrees.fit!(lnr_local, X, y);
b = OptimalTrees.score(lnr_local, X, y);

OptimalTrees.showinbrowser(lnr_local)
