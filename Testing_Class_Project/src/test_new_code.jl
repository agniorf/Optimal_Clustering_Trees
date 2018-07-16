import OptimalTrees
using Base.Test, RDatasets, Gadfly, Distances, Clustering
using DataFrames, MLDataUtils,Compat
using JLD
using Logging
Logging.configure(level=Logging.DEBUG)

dataset = dataset("cluster", "ruspini");
dataset_array = convert(Array{Float64}, dataset);
dataset_t = dataset_array';

K = 4;
srand(1234);
kmeans_result = kmeans(dataset_t, K);

assignments = kmeans_result.assignments;
dataset_full = DataFrame(hcat(dataset, assignments));
rename!(dataset_full, :x1, :kmean_assign)

# plot(dataset_full, x = :V2, y = :V3, color = :kmean_assign)
X = dataset_full[1:2]; y = dataset_full[3];
reload("OptimalTrees");
srand(890);

lnr_class = OptimalTrees.OptimalTreeClassifier(max_depth=3, cp=0.01, localsearch=true, criterion=:density, ls_num_tree_restarts =5, ls_random_seed = 890);
OptimalTrees.fit!(lnr_class, X, y)
OptimalTrees.showinbrowser(lnr_class)

(train_X, train_y), (valid_X, valid_y) = splitobs(shuffleobs((X, y)), at=0.7)
lnr = OptimalTrees.OptimalTreeClassifier(ls_random_seed=123, ls_num_tree_restarts =100)
grid = OptimalTrees.GridSearch(lnr, Dict(
    :max_depth => 2,
    :minbucket => [5],
    :criterion => [:density]
))
OptimalTrees.fit!(grid, train_X, train_y, valid_X, valid_y, validation_criterion=:silhouette)
OptimalTrees.showinbrowser(grid.best_lnr)

datafolderpath = "/Users/agni/Packages/Optimal_Clustering_Trees/data"
filepath = joinpath(datafolderpath, "localSearch_5.csv")
reload("OptimalTrees");

data = readtable(filepath, makefactors = true);
X = data[:,1:2]
y = data[:,3]

 s=1
 lnr2 = OptimalTrees.OptimalTreeClassifier(max_depth=2, cp=0.01, criterion=:density,
   	localsearch=true, ls_num_tree_restarts=20, ls_random_seed = s);
  OptimalTrees.fit!(lnr2, X, y)
  
OptimalTrees.showinbrowser(lnr2)


srand();
(train_X, train_y), (valid_X, valid_y) = splitobs(shuffleobs((X, y)), at=0.65)

reload("OptimalTrees");
lnr = OptimalTrees.OptimalTreeClassifier(ls_random_seed=1, ls_num_tree_restarts = 20)
grid = OptimalTrees.GridSearch(lnr, Dict(
    :max_depth => 2,
    :minbucket => [1],
    :criterion => [:density]), 
	autotune_cp = true)
OptimalTrees.fit!(grid, train_X, train_y, valid_X, valid_y, validation_criterion=:silhouette)
OptimalTrees.showinbrowser(grid.best_lnr)

reload("OptimalTrees");
lnr = OptimalTrees.OptimalTreeClassifier(ls_random_seed=1, ls_num_tree_restarts = 20, cp = 0.01)
grid = OptimalTrees.GridSearch(lnr, Dict(
    :max_depth => 2,
    :minbucket => [1],
    :criterion => [:density]), 
	autotune_cp = false)
OptimalTrees.fit!(grid, train_X, train_y, valid_X, valid_y, validation_criterion=:silhouette)
OptimalTrees.showinbrowser(grid.best_lnr)


filepath = joinpath(datafolderpath, "EuropeJobs.csv")
data = readtable(filepath, makefactors = true);
X = data[:,2:end]
y = data[:,1]

s =25
lnr2 = OptimalTrees.OptimalTreeClassifier(max_depth=4, cp=0.01, criterion=:density,
 	localsearch=true, ls_num_tree_restarts=3, ls_random_seed = s);
OptimalTrees.fit!(lnr2, X, y)

(train_X, train_y), (valid_X, valid_y) = splitobs(shuffleobs((X, y)), at=0.7)

#Logging.configure(level=OFF)
#reload("OptimalTrees");
lnr = OptimalTrees.OptimalTreeClassifier(ls_random_seed=1, ls_num_tree_restarts =10)
grid = OptimalTrees.GridSearch(lnr, Dict(
    :max_depth => 1:3,
    :minbucket => [1],
    :criterion => [:density],
))
OptimalTrees.fit!(grid, train_X, train_y, valid_X, valid_y, validation_criterion=:silhouette)


OptimalTrees.showinbrowser(grid.best_lnr)


lnr = OptimalTrees.OptimalTreeClassifier(max_depth=2, localsearch=true, ls_num_tree_restarts =5, ls_random_seed = 890);
grid = OptimalTrees.GridSearch(lnr, Dict(:criterion => [:silhouette]))
OptimalTrees.fit!(grid, X, y, validation_criterion=:density)

lnr_greedy = OptimalTrees.OptimalTreeClassifier(max_depth=3, cp=0, localsearch=false, criterion=:cluster);
OptimalTrees.fit!(lnr_greedy, X, y)
OptimalTrees.showinbrowser(lnr_greedy)

lnr2 = OptimalTrees.OptimalTreeClassifier(max_depth=3, cp=0, localsearch=true, criterion=:cluster, ls_num_tree_restarts=10);
OptimalTrees.fit!(lnr2, X, y)
OptimalTrees.showinbrowser(lnr2)


#First we would like to read the dataset
#Set the directory
datafolderpath = "/Users/Dropbox\ \(Personal\)/git/Optimal_Clustering_Trees/Testing_Class_Project/data"
# resultsfolderpath = "/Users/agni/Packages/Optimal_Clustering_Trees/Testing_Class_Project/results"

filepath = joinpath(datafolderpath, "Lsun.csv")
resultspath = joinpath(datafolderpath, "Lsun.jld")

data = readtable(filepath, makefactors = true);

oracle_assignments = data[:,:label]

lnr = OptimalTrees.OptimalTreeClassifier(max_depth=5, cp=0.01, localsearch=true, criterion=:cluster, ls_num_tree_restarts=100)
srand(100)
OptimalTrees.fit!(lnr, data[:,1:(end-1)], data[:,:label])

@save "WingNut.jld" lnr

@load "WingNut.jld" lnr

OptimalTrees.showinbrowser(lnr)
#Find in which node each observation is mapped
leaf_assignment = OptimalTrees.apply(lnr, data[:,1:(end-1)])

