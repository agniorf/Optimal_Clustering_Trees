using RDatasets, MLDataUtils
using Clustering
using Distances

data_raw = dataset("cluster", "ruspini");
data_t= Array{Float64}(data_raw)';

srand(2);
kmeans_result = kmeans(data_t, 4);
assignments = kmeans_result.assignments;
colors = recode(assignments, 4=> "blue", 3 => "red", 2=> "blue", 1=> "yellow")

X = DataFrame(hcat(colors, data_raw[:Y]));
pool!(X, :x1);
y = assignments;

num_tree_restarts = 1; seed = 2; complexity = 0; maxdepth = 3; min_bucket = 1; cr = :silhouette;

lnr = OptimalTrees.OptimalTreeClassifier(ls_num_tree_restarts = num_tree_restarts, ls_random_seed = seed, 
	cp = complexity, max_depth = maxdepth, minbucket = min_bucket, criterion = cr, ls_warmstart_criterion = cr);

OptimalTrees.fit!(lnr, X, y);