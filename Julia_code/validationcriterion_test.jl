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

ls_data = readtable("../data/localSearch_1.csv");
X = ls_data[1:2]; y = ls_data[3];

europe_data = readtable("../data/EuropeJobs.csv");
X = europe_data[2:end]; y = europe_data[1];

(train_X, train_y), (valid_X, valid_y) = splitobs(shuffleobs((X, y)), at=0.67);

#### WITH GRID SEARCH


# reload("OptimalTrees");
s = 22;
lnr_grid = OptimalTrees.OptimalTreeClassifier(ls_num_tree_restarts = 100, ls_random_seed = s, cp = 0.0,
	criterion = :density, show_progress_bar=true);

grid = OptimalTrees.GridSearch(lnr_grid, Dict(
    :max_depth => 1:5),
    autotune_cp = false);

OptimalTrees.fit!(grid, train_X, train_y, valid_X, valid_y, validation_criterion = :silhouette)
OptimalTrees.showinbrowser(grid.best_lnr)

reload("OptimalTrees");
# srand(890);

### WITHOUT GRID SEARCH
s = 25;
lnr2 = OptimalTrees.OptimalTreeClassifier(max_depth=3, cp=0.0, criterion=:density,
 	localsearch=true, ls_num_tree_restarts=3, ls_random_seed = s);
OptimalTrees.fit!(lnr2, X, y)
OptimalTrees.showinbrowser(lnr2)


# lnr_grid = OptimalTrees.OptimalTreeClassifier(localsearch=false, cp = 0, criterion = :density, show_progress_bar=true);

# grid = OptimalTrees.GridSearch(lnr_grid, Dict(
#     :max_depth => 1:3));

# OptimalTrees.fit!(grid, train_X, train_y, valid_X, valid_y, validation_criterion = :silhouette)
# OptimalTrees.showinbrowser(grid.best_lnr)



# plot(dataset_full, x = :V2, y = :V3, color = :kmean_assign)


################## ERRORS

reload("OptimalTrees");

# s = 25;
# X = europe_data[2:end]; y = europe_data[1];
# lnr2 = OptimalTrees.OptimalTreeClassifier(max_depth=3, cp=0.01, criterion=:density,
#  	localsearch=true, ls_num_tree_restarts=2, ls_random_seed = s);
# OptimalTrees.fit!(lnr2, X, y)


reload("OptimalTrees");

lnr_grid = OptimalTrees.OptimalTreeClassifier(ls_num_tree_restarts = 5, ls_random_seed = s, cp = .05,
	criterion = :density, show_progress_bar=true);

grid = OptimalTrees.GridSearch(lnr_grid, Dict(
    :max_depth => 1:3),
    autotune_cp = false);

OptimalTrees.fit!(grid, train_X, train_y, valid_X, valid_y, validation_criterion = :silhouette)
# OptimalTrees.showinbrowser(grid.best_lnr)


ls_data = readtable("../data/localSearch_1.csv");
X = ls_data[1:2]; y = ls_data[3];

srand(1);
(train_X, train_y), (valid_X, valid_y) = splitobs(shuffleobs((X, y)), at=0.7);

lnr = OptimalTrees.OptimalTreeClassifier(ls_random_seed=2, ls_num_tree_restarts =20)
grid = OptimalTrees.GridSearch(lnr, Dict(
    :max_depth => 1:2,
    :minbucket => [5],
    :criterion => [:density]
));
OptimalTrees.fit!(grid, train_X, train_y, valid_X, valid_y, validation_criterion=:silhouette);
OptimalTrees.showinbrowser(grid.best_lnr)

