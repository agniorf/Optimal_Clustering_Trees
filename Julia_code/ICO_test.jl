using DataFrames, MLDataUtils
using Clustering, Distances
using RDatasets
using OptimalTrees
using Gadfly
using BenchmarkTools


# dataset = readtable("../Experiments/Experiment_1/data/Lsun.csv"); 
# dataset = readtable("../Experiments/Experiment_1/data/Lsun.csv"); 
# dataset = readtable("../Testing_Class_Project/data/TwoDiamonds.csv")
# dataset = readtable("../data/localSearch_6.csv");
# X = dataset[1:end-1]; y = dataset[end];

data = dataset("cluster", "ruspini");
data_array = convert(Array{Float64}, data);
data_t = data_array';

K = 3;
srand(1234);
kmeans_result = kmeans(data_t, K);

assignments = kmeans_result.assignments;
data_full = DataFrame(hcat(data, assignments));
rename!(data_full, :x1, :kmean_assign);
# plot(dataset_full, x = :V2, y = :V3, color = :kmean_assign)

X = data_full[1:2]; y = data_full[end];



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
@btime OptimalTrees.fit!(lnr_local, X, y);
b = OptimalTrees.score(lnr_local, X, y);

OptimalTrees.showinbrowser(lnr_local)
