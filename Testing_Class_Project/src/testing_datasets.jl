import OptimalTrees
using Base.Test, RDatasets, Gadfly, Distances, Clustering
using DataFrames, MLDataUtils,Compat
using JLD

#First we would like to read the dataset
#Set the directory
datafolderpath = "/Users/agni/Packages/Optimal_Clustering_Trees/Testing_Class_Project/data"
resultsfolderpath = "/Users/agni/Packages/Optimal_Clustering_Trees/Testing_Class_Project/results"

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

