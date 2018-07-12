using DataFrames
using RDatasets


# dataset = dataset("cluster", "ruspini");
# dataset_array = convert(Array{Float64}, dataset);
# dataset_t = dataset_array';

# K = 4;
# srand(1234);
# kmeans_result = kmeans(dataset_t, K);

# assignments = kmeans_result.assignments;
# dataset_full = DataFrame(hcat(dataset, assignments));
# rename!(dataset_full, :x1, :kmean_assign)

# X = dataset_full[1:2]; y = dataset_full[:kmean_assign];

rusp = dataset("cluster", "ruspini");
X = rusp[1:2]; y = ones(size(X,1));

### WITHOUT GRID SEARCH
reload("OptimalTrees")
s = 25;

ls_data = readtable("../data/localSearch_6.csv");
X = ls_data[1:2]; y = ls_data[3];

lnr2 = OptimalTrees.OptimalTreeClassifier(max_depth=2, cp=0, criterion=:silhouette,
 	localsearch=true, ls_num_tree_restarts=30, ls_random_seed = s);
OptimalTrees.fit!(lnr2, X, y)
OptimalTrees.showinbrowser(lnr2)

# europe_data = readtable("../data/EuropeJobs.csv");
# X = europe_data[2:end]; y = europe_data[1];

# ls_data = readtable("../data/localSearch_6.csv");
# X = ls_data[1:2]; y = ones(size(ls_data, 1));

#### WITH GRID SEARCH


# reload("OptimalTrees");
s = 22;
(train_X, train_y), (valid_X, valid_y) = splitobs(shuffleobs((X, y)), at=0.67);

lnr_grid = OptimalTrees.OptimalTreeClassifier(ls_num_tree_restarts = 50, ls_random_seed = s, cp = 0.0,
	criterion = :silhouette, show_progress_bar=true);

grid = OptimalTrees.GridSearch(lnr_grid, Dict(
    :max_depth => 1:3),
    autotune_cp = false);

OptimalTrees.fit!(grid, train_X, train_y, valid_X, valid_y, validation_criterion = :silhouette)
OptimalTrees.showinbrowser(grid.best_lnr)

reload("OptimalTrees");
# srand(890);



# lnr_grid = OptimalTrees.OptimalTreeClassifier(localsearch=false, cp = 0, criterion = :density, show_progress_bar=true);

# grid = OptimalTrees.GridSearch(lnr_grid, Dict(
#     :max_depth => 1:3));

# OptimalTrees.fit!(grid, train_X, train_y, valid_X, valid_y, validation_criterion = :silhouette)
# OptimalTrees.showinbrowser(grid.best_lnr)



# plot(dataset_full, x = :V2, y = :V3, color = :kmean_assign)


################## ERRORS

reload("OptimalTrees");

# s = 25;
X = europe_data[2:end]; y = europe_data[1];
lnr2 = OptimalTrees.OptimalTreeClassifier(max_depth=5, cp=0.2, criterion=:density,
 	localsearch=true, ls_num_tree_restarts=100, ls_random_seed = 100);
OptimalTrees.fit!(lnr2, train_X, train_y)


reload("OptimalTrees");


X = europe_data[2:end]; y = europe_data[1];
(train_X, train_y), (valid_X, valid_y) = splitobs(shuffleobs((X, y)), at=0.7);

lnr_grid = OptimalTrees.OptimalTreeClassifier(ls_num_tree_restarts = 5, ls_random_seed = s, cp = .15,
	criterion = :density, show_progress_bar=true);

grid = OptimalTrees.GridSearch(lnr_grid, Dict(
    :max_depth => 1:3),
    autotune_cp = false);

OptimalTrees.fit!(grid, train_X, train_y, valid_X, valid_y, validation_criterion = :silhouette)
# OptimalTrees.showinbrowser(grid.best_lnr)



############################# USE THIS $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

ls_data = readtable("../data/localSearch_3.csv");
X = ls_data[1:2]; y = ls_data[2];


reload("OptimalTrees");
s = 25;
lnr2 = OptimalTrees.OptimalTreeClassifier(max_depth=3, cp=0.2, criterion=:density,
 	localsearch=true, ls_num_tree_restarts=3, ls_random_seed = s);
OptimalTrees.fit!(lnr2, X, y)
OptimalTrees.showinbrowser(lnr2)



ls_data = readtable("../data/localSearch_3.csv");


reload("OptimalTrees")

s = 10;
srand(s);
X = ls_data[1:2]; y = ones(size(X,1))
(train_X, train_y), (valid_X, valid_y) = splitobs(shuffleobs((X, y)), at=0.7);

lnr = OptimalTrees.OptimalTreeClassifier(ls_random_seed=s, ls_num_tree_restarts =20, minbucket = 5, criterion = :density)
grid = OptimalTrees.GridSearch(lnr, Dict(
    :max_depth => 1:3),
	autotune_cp = true);
OptimalTrees.fit!(grid, train_X, train_y, valid_X, valid_y, validation_criterion=:silhouette);
OptimalTrees.showinbrowser(grid.best_lnr)

