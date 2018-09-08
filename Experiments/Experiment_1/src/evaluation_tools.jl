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

function eval_kmeans(X, lnr, k_range, seed, cr)

	score_dict = Dict{Int64,Float64}()
	assignments_dict = Dict{Int64,Array{Int64}}()
	X_t = Array{Float64}(X)'

	for k in k_range
		srand(seed)
		kmeans_result = kmeans(X_t, k);
		assignments = kmeans_result.assignments;
		# fullresult = DataFrame(hcat(X, assignments));
		score_dict[k] = cluster_score(lnr, assignments, cr)
		assignments_dict[k] = assignments
	end

	bestk = collect(keys(score_dict))[indmax(values(score_dict))]

	# return score, assignments
	return score_dict, assignments_dict, bestk

end

function silhouette_score(lnr, assignments)
	K = length(unique(assignments));

	distance_matrix = lnr.prb_.data.features.distance_matrix;

	assign_matrix = hcat(collect(1:size(assignments,1)), assignments);
	
	if length(unique(assign_matrix[:,2])) == 1
		score = -2
	else
		# Bring in assignments and reorder to be 1-k
		clustdict = Dict{Int64, Int64}()
		k = 1
		for clustindex in unique(assign_matrix[:,2])
		  clustdict[clustindex] = k
		  k += 1
		end

		# Reassign to new cluster indices
		for i in 1:size(assign_matrix,1)
		  assign_matrix[i,2] = clustdict[assign_matrix[i,2]]
		end

		assignments_ordered = sortrows(assign_matrix, by=x->x[1])[:,2]

		counts = Int64[]
		for i in sort(unique(assignments_ordered))
		  push!(counts, count(assignments_ordered.==i))
		end

		sil = silhouettes(assignments_ordered, counts, distance_matrix)

		# Find average silhouette score of obs in this leaf
		score = mean(sil)
		# println("SCORE: ", -score)
	end

  return score
end

function dunn_score(lnr, assignments)

	K = length(unique(assignments));

	distance_matrix = lnr.prb_.data.features.distance_matrix;


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
	    println("The minimum separation is :", minimum_separation)
        println("The maximum_diameter is :", maximum_diameter)
	    score = minimum_separation/maximum_diameter
	end

    return score

end

function robustdunn_score(lnr, assignments)

	λ = .2
	K = length(unique(assignments));
	distance_matrix = lnr.prb_.data.features.distance_matrix;

	assign_df = DataFrame(hcat(collect(1:size(assignments,1)), assignments));

	n = size(assign_df,1)
	pairwise_n = n*(n-1)/2

	if K == 1
		score = -10000000
	else
	# Find top λ% of diameters and sum them
	clust_dict = Dict{Int64, Array{Int64,1}}()

	intraclust_distances = []
	maximum_diameter = 0
	intra_n = 0
	for clust in unique(assignments)
		# println("Cluster number = ", clust)
		index_list = assign_df[assign_df[:x2] .== clust, :x1];
		clust_dict[clust] = index_list;
		cluster_distances = distance_matrix[index_list, index_list];
		intraclust_distances = vcat(intraclust_distances, collect(Iterators.flatten(cluster_distances)));
		intra_n += size(index_list,1)*(size(index_list,1)-1)/2
		# println("Intra_n = $(intra_n); elements = $(size(index_list,1))")
	end

	intra_cutoff = convert(Int64, round(λ*intra_n/2));
	sorted_intraclust = sort(unique(intraclust_distances), rev = true);
	sum_high_diams = sum(sorted_intraclust[1:intra_cutoff]);

	betweenclust_distances = []
	between_n = pairwise_n - intra_n
	minimum_separation = 1000000
	for c1 in unique(assignments), c2 in setdiff(unique(assignments),c1)
	  	if c1 < c2
	  		cluster_distances = distance_matrix[clust_dict[c1], clust_dict[c2]]
	    	betweenclust_distances = vcat(betweenclust_distances, collect(Iterators.flatten(cluster_distances)));
	  	end
	end

	between_cutoff = convert(Int64, round(λ*between_n/2));
	sorted_betweenclust = sort(unique(betweenclust_distances));
	sum_low_seps = sum(sorted_betweenclust[1:between_cutoff]);

	score = (sum_low_seps/between_cutoff)/(sum_high_diams/intra_cutoff)
	end

	return score
end



function cluster_score(lnr, assignments, cr)
	if cr == :silhouette
		score = silhouette_score(lnr, assignments)
	elseif cr == :dunnindex
		score = dunn_score(lnr, assignments)
	elseif cr == :robustdunn
		score = robustdunn_score(lnr, assignments)
	else score = -10
	end
	return score
end
