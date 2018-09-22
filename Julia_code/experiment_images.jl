using JLD
using OptimalTrees

datasets = readdir("~/research/Optimal_Clustering_Trees/Experiments/Experiment_1/data/");
# criter=[:silhouette,:dunnindex]
criter=[:robustdunn];
clust_method=["localsearch","greedy"];

;));

for i in 1:length(datalistcriterionpairs)
	(d, c, m) = datalistcriterionpairs[i]

	result = load("lnr-$d-$c-$m-lnr.jld");

	lnr = result["lnr"]
	OptimalTrees.writedot("$d-$c-$m.dot", lnr);
	run(`dot -Tpng -o $d-$c-$m.png $d-$c-$m.dot`);
end