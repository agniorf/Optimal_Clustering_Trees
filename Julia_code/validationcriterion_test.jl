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

ls_data = readtable("../data/localSearch_2.csv");
X = ls_data[1:2]; y = ls_data[3];

# europe_data = readtable("../data/EuropeJobs.csv");
# X = europe_data[2:end]; y = europe_data[1];

(train_X, train_y), (valid_X, valid_y) = splitobs(shuffleobs((X, y)), at=0.67);

#### WITH GRID SEARCH


# reload("OptimalTrees");
s = 1234;
lnr_grid = OptimalTrees.OptimalTreeClassifier(ls_num_tree_restarts = 10, ls_random_seed = s, criterion = :density, show_progress_bar=true);

grid = OptimalTrees.GridSearch(lnr_grid, Dict(
    :max_depth => 1:3),
    autotune_cp = true);

OptimalTrees.fit!(grid, train_X, train_y, valid_X, valid_y, validation_criterion = :silhouette)
OptimalTrees.showinbrowser(grid.best_lnr)

# reload("OptimalTrees");
# srand(890);

### WITHOUT GRID SEARCH
lnr2 = OptimalTrees.OptimalTreeClassifier(max_depth=3, cp=0.0, localsearch=true, criterion=:density, ls_num_tree_restarts=10);
OptimalTrees.fit!(lnr2, X, y)
OptimalTrees.showinbrowser(lnr2)



# plot(dataset_full, x = :V2, y = :V3, color = :kmean_assign)