using DataFrames
using RDatasets
using Distances
using Clustering


rusp = dataset("cluster", "ruspini"); X = rusp[1:2]; y = ones(size(X,1)); truelabels = false;
ruspini_t = convert(Array{Float64}, rusp)';
rusp_kmeans = kmeans(ruspini_t, 4);

distance_matrix = pairwise(Euclidean(), ruspini_t);
assignments = rusp_kmeans.assignments;

function dunn_index(distance_matrix, assignments)
	K = length(unique(assignments));
	distances = deepcopy(distance_matrix);
	assign_df = DataFrame(hcat(collect(1:size(assignments,1)), assignments));

	# Find maximum diameter and store indices for each assignment
	clust_dict = Dict{Int64, Array{Int64,1}}()
	maximum_diameter = 0
	# for clust in 1:K
	# 	index_list = assign_df[assign_df[:x2] .== clust, :x1];
	# 	max_dist = maximum(distance_matrix[index_list, index_list])
	# 	maximum_diameter = max(maximum_diameter, max_dist)
	# 	clust_dict[clust] = index_list
	# end

	# minimum_separation = 1000000
	# for c1 in 1:K, c2 in 1:(c1-1)
	# 	min_dist = minimum(distance_matrix[clust_dict[c1], clust_dict[c2]])
	# 	minimum_separation = min(minimum_separation, min_dist)
	# end

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


	println("maximum diameter = ", maximum_diameter)
	println("minimum separation = ", minimum_separation)

	return maximum_diameter/minimum_separation
end