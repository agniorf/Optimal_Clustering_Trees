using JLD
using OptimalTrees

datasets = readdir("/home/hwiberg/research/Optimal_Clustering_Trees/Experiments/Experiment_1/data/");
criter=[:silhouette,:dunnindex]
# criter=[:robustdunn];
clust_method=["localsearch","greedy"];

 datalistcriterionpairs =collect(Iterators.product(datasets,criter,clust_method));


for i in 1:length(datalistcriterionpairs)
	(d, c, m) = datalistcriterionpairs[i]

	result = load("lnr-$d-$c-$m-lnr.jld");

	lnr = result["lnr"]
	OptimalTrees.writedot("$d-$c-$m.dot", lnr);
	run(`dot -Tpng -o $d-$c-$m.png $d-$c-$m.dot`);
end