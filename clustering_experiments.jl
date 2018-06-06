# using RDatasets, MLDataUtils
# using Clustering
# using Distances

# import OptimalTrees

# seed = 1

# # Choose dataset
# # data_raw = dataset("cluster", "xclara");
# data_raw = dataset("cluster", "ruspini");
# X = data_raw; 
# # y = ones(size(data_raw, 1))
# y = data_raw[1];

reload("OptimalTrees")

lnr2 = OptimalTrees.OptimalTreeClassifier(max_depth=4, cp=0.01, criterion = :cluster, localsearch = false);
OptimalTrees.fit!(lnr2, X, y)
# OptimalTrees.showinbrowser(lnr2)

leaf_cnt = 0
for i in 1:lnr2.tree_.node_count
	if lnr2.tree_.nodes[i].lower_child == -2
		leaf_cnt += 1
	end 
end

lnrscore = -lnr2.tree_.nodes[1].raw_error

@save ".results/learners_xclara.jld" lnr2

# println("Silhouette Score (Opt Clustering, K = $(leaf_cnt)): ", lnrscore)

# ######### K Means ##########

function eval_kmeans(X, k, seed)
	println("Running k means for k = $(k)")
	srand(seed)
	X_t = Array{Float64}(X)'

	dist_matrix = pairwise(Euclidean(), X_t);

	kmeans_result = kmeans(X_t, k);
	assignments = kmeans_result.assignments;
	fullresult = DataFrame(hcat(X, assignments));

	counts = Int64[]
	for i in sort(unique(assignments))
		push!(counts, count(assignments.==i))
	end

	dist_matrix = pairwise(Euclidean(), X_t)

	return mean(silhouettes(assignments, counts, dist_matrix))

end

for k = 2:5
	result = eval_kmeans(X, k, 1)
	println("K means Score (K = $(k)): ", result)
end