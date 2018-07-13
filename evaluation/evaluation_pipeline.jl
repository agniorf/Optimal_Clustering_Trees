using DataFrames, MLDataUtils
using Clustering, Distances
using RDatasets
using OptimalTrees
# reload("OptimalTrees")

include("evaluation_tools.jl")

####### STEP 1: PULL IN DATA. Set seed and choose dataset (X, y). If unsupervised, set y to be all ones
s = 2;
gridsearch = false;
cr = :dunnindex;

# # Choose a dataset to bring in
# rusp = dataset("cluster", "ruspini"); X = rusp[1:2]; y = ones(size(X,1)); truelabels = false;

# europe_data = readtable("../data/EuropeJobs.csv"); X = europe_data[2:end];  y = europe_data[1]); truelabels = false;

# ls_data = readtable("../data/localSearch_7.csv"); X = ls_data[1:2];  y = ls_data[3]; truelabels = false;

# xclara_data = dataset("cluster", "xclara"); X = xclara_data[1:2]; y = ones(size(X,1)); truelabels = false;

# lsun_data = readtable("../Testing_Class_Project/data/lsun.csv"); X = lsun_data[1:2]; y = lsun_data[3]; truelabels = true;

wingnut_data = readtable("../Testing_Class_Project/data/wingnut.csv"); 
# wingnut_data = wingnut_data_raw[rand(collect(1:size(wingnut_data_raw,1)),200),:]; 
X = wingnut_data[1:2]; y = wingnut_data[3]; truelabels = true; 

# # Split into training and validation sets
srand(s)
(train_X, train_y), (valid_X, valid_y) = splitobs(shuffleobs((X, y)), at=0.67);

####### STEP 2: OPTCLUST. Run clustering with grid search (default to autotune cp and run max_depth 1-5)

println("Running Optimal Trees")

lnr = OptimalTrees.OptimalTreeClassifier(ls_num_tree_restarts = 5, ls_random_seed = s, cp = 0.0, max_depth = 3,
	minbucket = 1, criterion = cr, ls_warmstart_criterion = cr, show_progress_bar=true);
OptimalTrees.fit!(lnr, X, y);

lnr_greedy = OptimalTrees.OptimalTreeClassifier(localsearch = false, cp = 0.0, max_depth = 3,
	minbucket = 1, criterion = cr, show_progress_bar=true);
OptimalTrees.fit!(lnr_greedy, X, y);

# if gridsearch
# 	lnr_grid = OptimalTrees.OptimalTreeClassifier(ls_num_tree_restarts = 100, ls_random_seed = s, cp = 0.0,
# 		minbucket = 1, criterion = :silhouette, show_progress_bar=true);

# 	grid = OptimalTrees.GridSearch(lnr_grid, Dict(:max_depth => 1:4), autotune_cp = false);

# 	OptimalTrees.fit!(grid, train_X, train_y, valid_X, valid_y, validation_criterion = :silhouette);
# 	lnr = grid.best_lnr;
# else 
# end

OptimalTrees.showinbrowser(lnr)
OptimalTrees.showinbrowser(lnr_greedy)

leaf_cnt = leafcount(lnr) ## Defined in tools file
optclust_depth = grid.best_params[:max_depth];
optclust_assignments = OptimalTrees.apply(lnr, X);
optclust_score = OptimalTrees.score(lnr, X, y, criterion = cr);

####### STEP 3: KMEANS. Run k means with the chosen depth, and in the neighborhood of the tree

println("Running K means")
# Determine range for k based on leaves of OptClust result
min_k = max(leaf_cnt - 2, 2); max_k = leaf_cnt + 2;

# Return a dictionary of scores and assignments for each k, as well as the best k value (max score)
kmeans_scoredict, kmeans_assignmentdict, kmeans_best = eval_kmeans(X, min_k:max_k, s); ## Defined in tools file
ari_optclust_kmeans = randindex(optclust_assignments, kmeans_assignmentdict[leaf_cnt])[1];

##### STEP 5: If ground truth available, compare
if truelabels
	true_assignments = Array{Int64}(y)
	true_k = length(unique(y));
	true_silhouette = silhouette_score(X, true_assignments)
	ari_true_kmeans = randindex(true_assignments, kmeans_assignmentdict[true_k])[1]
	if true_k == leaf_cnt 
		ari_true_optclust = randindex(true_assignments, optclust_assignments)[1]
	else ari_true_optclust = -10
	end
else true_assignments = -10; true_k = -10; true_silhouette = -10; ari_true_kmeans = -10;
end


###### STEP 4: COMPARE METHODS. 

## What are the chosen k and scores: 
println("OptClust: K = $(leaf_cnt); Score = $(optclust_score)");
println("Kmeans (best K): K = $(kmeans_best); Score = $(kmeans_scoredict[kmeans_best])");
println("Kmeans (OptClust K): K = $(leaf_cnt); Score = $(kmeans_scoredict[leaf_cnt])");
println("True Assignments: K = $(true_k); Score = $(true_silhouette)")

### Compare agreement in assignments
println("Adjusted Rand Index - OptClust vs. Kmeans (K = $(leaf_cnt)): ", ari_optclust_kmeans);
println("Adjusted Rand Index - True vs. Kmeans (K = $(true_k)): ", ari_true_kmeans);
println("Adjusted Rand Index - OptClust vs. True (if k matches)): ", ari_true_optclust);

### Save results in an array to paste into Excel file 
results = [s, optclust_depth, leaf_cnt, kmeans_best, true_k, 
	optclust_score, kmeans_scoredict[kmeans_best], kmeans_scoredict[leaf_cnt], true_silhouette,
	ari_optclust_kmeans, ari_true_kmeans, ari_true_optclust];
println("[seed; depth; K (optclust (kc), kmeans, true); 
	score (optclust, kmeans (best k), kmeans (kc), true); 
	ARI (optclust vs. kmeans, true vs. kmeans, true vs. optclust)]")
# for i in 1:size(results,1) print("$(results[i])   ")  end
println(results')

println(lnr)


