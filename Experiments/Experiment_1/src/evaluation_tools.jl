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

function eval_kmeans(X, k_range, seed, cr)

	score_dict = Dict{Int64,Float64}()
	assignments_dict = Dict{Int64,Array{Int64}}()
	X_t = Array{Float64}(X)'

	for k in k_range
		srand(seed)
		kmeans_result = kmeans(X_t, k);
		assignments = kmeans_result.assignments;
		# fullresult = DataFrame(hcat(X, assignments));
		if cr == :silhouette
			score_dict[k] = silhouette_score(X, assignments)
		elseif cr == :dunnindex
			score_dict[k] = dunn_score(X, assignments)
		else score_dict[k] = 10
		end
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

function dunn_score(X, assignments)

	K = length(unique(assignments));

	X_t = Array{Float64}(X)'
	distance_matrix = pairwise(Euclidean(), X_t);


	assign_df = DataFrame(hcat(collect(1:size(assignments,1)), assignments));

  
	if K == 1
		score = 0
	else
    # Find maximum diameter and store indices for each assignment
    clust_dict = Dict{Int64, Array{Int64,1}}()
    maximum_diameter = 0
    for clust in unique(assignments)
      index_list = assign_df[assign_df[:x2] .== clust, :x1];
      max_dist = maximum(distance_matrix[index_list, index_list])
      maximum_diameter = max(maximum_diameter, max_dist)
      clust_dict[clust] = index_list
      if index_list == []
        println("ALERT!! EMPTY INDEX LIST")
      end
    end

    minimum_separation = 1000000
    for c1 in unique(assignments), c2 in setdiff(unique(assignments),c1)
        min_dist = minimum(distance_matrix[clust_dict[c1], clust_dict[c2]])
        minimum_separation = min(minimum_separation, min_dist)
    end

    score = minimum_separation/maximum_diameter
  end

