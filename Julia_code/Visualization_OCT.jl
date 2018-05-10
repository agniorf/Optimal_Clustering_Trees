using DataFrames, OptimalTrees
using RDatasets, MLDataUtils, JLD

;cd "/Users/agni/Dropbox (Personal)/Clustering/data/FCPS/05Results"

data = readtable("Atom_hierarchical_k_means_k_2.csv", makefactors = true);

X = data[:,1:3]
k_means = data[:,:k_means_cluster]
hierarch = data[:,:hierarch_cluster]
true_l = data[:,:true_label]

data = readtable("Lsun_hierarchical_k_means_k_3.csv", makefactors = true);

X = data[:,1:2]
k_means = data[:,:k_means_cluster]
hierarch = data[:,:hierarch_cluster]
true_l = data[:,:true_label]

grid_k_means, MisC_k_means = fitOptTree(X, k_means)
OptimalTrees.showinbrowser(grid_k_means.best_lnr)

grid_hierarch, MisC_hierarch = fitOptTree(X, hierarch)
OptimalTrees.showinbrowser(grid_hierarch.best_lnr)

grid_true_l, MisC_true_l = fitOptTree(X, true_l)
OptimalTrees.showinbrowser(grid_true_l.best_lnr)


function fitOptTree(X,Y;
                    seed::Int=123,
                    max_depth_range::Union{Int,Range}=1:5,
                    minbucket_range::Union{Real,Range}=:10:80,
                    fitcv::Bool=false,
                    missingdatamode::Symbol=:none)
  srand(seed)
  (big_X, big_Y), (test_X, test_Y) = stratifiedobs((X, Y), p=0.75);
  (train_X, train_Y), (valid_X, valid_Y) = stratifiedobs((big_X, big_Y), p=0.67);
  lnr = OptimalTrees.OptimalTreeClassifier(ls_random_seed = 1,
                                           missingdatamode = missingdatamode)
  lnr.treat_unknown_categoric_missing = true
  tree_params = Dict{Symbol,Any}(
    :max_depth => max_depth_range,
    :minbucket => minbucket_range,
    :criterion => [:entropy]
  )
  grid = OptimalTrees.GridSearch(lnr, tree_params)
  if fitcv
    OptimalTrees.fit_cv!(grid, big_X, big_Y,
                    validation_criterion=:misclassification, verbose=true)
  else
    OptimalTrees.fit!(grid, train_X, train_Y, valid_X, valid_Y,
                    validation_criterion=:misclassification, verbose=true)
  end
  MisC = OptimalTrees.score(grid.best_lnr, test_X, test_Y, criterion=:misclassification)
  #AUC = OptimalTrees.score(grid.best_lnr, test_X, test_Y, criterion=:auc)
  return grid, MisC
end
