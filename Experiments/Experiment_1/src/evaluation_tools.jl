Pkg.add("HttpCommon")
using DataFrames, MLDataUtils
using Clustering, Distances
using RDatasets
using OptimalTrees, JLD

# Evaluating Clusters 
function leafcount(lnr)
	leaf_cnt = 0
	for i in 1:lnr.tree_.node_count
		if lnr.tree_.nodes[i].lower_child == -2
			leaf_cnt += 1
		end 
	end
	return leaf_cnt
end

# Running K means for several k
function eval_kmeans(X, k_range, seed)

	score_dict = Dict{Int64,Float64}()
	assignments_dict = Dict{Int64,Array{Int64}}()
	X_t = Array{Float64}(X)'

	for k in k_range
		srand(seed)
		kmeans_result = kmeans(X_t, k);
		assignments = kmeans_result.assignments;
		# fullresult = DataFrame(hcat(X, assignments));

		score_dict[k] = silhouette_score(X, assignments)
		assignments_dict[k] = assignments
	end

	bestk = collect(keys(score_dict))[indmax(values(score_dict))]

	# return score, assignments
	return score_dict, assignments_dict, bestk

end

function silhouette_score(X, assignments)
	X_t = Array{Float64}(X)'
	dist_matrix = pairwise(Euclidean(), X_t);

	counts = Int64[]
	for i in sort(unique(assignments))
		push!(counts, count(assignments.==i))
	end

	return mean(silhouettes(assignments, counts, dist_matrix))
end


