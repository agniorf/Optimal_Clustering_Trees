using RDatasets, MLDataUtils
using Clustering
using Gadfly

# import OptimalTrees

# iris = dataset("datasets", "iris");
# X = iris[1:2]; y = iris[5];
# srand(1);
# (big_X, big_y), (test_X, test_y) = stratifiedobs((X, y), p=0.75);
# (train_X, train_y), (valid_X, valid_y) = stratifiedobs((big_X, big_y), p=0.67);

# lnr = OptimalTrees.OptimalTreeClassifier(max_depth=2, cp=0.01, criterion = :gini);
# OptimalTrees.fit!(lnr, train_X, train_y)

# reload("OptimalTrees")
# lnr2 = OptimalTrees.OptimalTreeClassifier(max_depth=2, cp=0.01, criterion = :cluster, localsearch = false);
# OptimalTrees.fit!(lnr2, train_X, train_y)

######### RUSPINI

using RDatasets, MLDataUtils
using Clustering
using Gadfly

# datafolderpath = "/Users/hollywiberg/Dropbox (Personal)/git/Optimal_Clustering_Trees/Testing_Class_Project/data"

# filepath = joinpath(datafolderpath, "Lsun.csv");
# dataset = readtable(filepath, makefactors = true);

dataset = dataset("cluster", "ruspini");
dataset_array = convert(Array{Float64}, dataset);
dataset_t = dataset_array';

K = 3;
srand(1234);
kmeans_result = kmeans(dataset_t, K);

assignments = kmeans_result.assignments;
dataset_full = DataFrame(hcat(dataset, assignments));
rename!(dataset_full, :x1, :kmean_assign)
dataset_full[:binary] = ifelse.((dataset_full[:kmean_assign] .== 3) | (dataset_full[:kmean_assign] .== 4), "A", "B")

# plot(dataset_full, x = :V2, y = :V3, color = :kmean_assign)

X = dataset_full[1:2]; y = dataset_full[end];

reload("OptimalTrees");
srand(890);

lnr_greedy = OptimalTrees.OptimalTreeClassifier(max_depth=3, cp=0, localsearch=false, criterion=:cluster);
OptimalTrees.fit!(lnr_greedy, X, y)
OptimalTrees.showinbrowser(lnr_greedy)

lnr2 = OptimalTrees.OptimalTreeClassifier(max_depth=3, cp=0, localsearch=true, criterion=:cluster, ls_num_tree_restarts=10);
OptimalTrees.fit!(lnr2, X, y)
OptimalTrees.showinbrowser(lnr2)

# writetable("ruspini.csv", dataset)

# plot(data, x = :x1, y = :x2, color = :x3)