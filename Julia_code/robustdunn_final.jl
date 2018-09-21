
##### ROBUST DUNN

function raw_error(assignments::Array{Int64,1}, distance_matrix::Array{Float64,2}, ::RobustDunn)
  λ = .01

  # Bring in distances
  K = length(unique(assignments))

  distances = deepcopy(distance_matrix);
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

    intra_cutoff = convert(Int64, ceil(λ*intra_n));
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

    between_cutoff = convert(Int64, ceil(λ*between_n));
    sorted_betweenclust = sort(unique(betweenclust_distances));
    sum_low_seps = sum(sorted_betweenclust[1:between_cutoff]);


    score = (sum_low_seps/between_cutoff)/(sum_high_diams/intra_cutoff)
  end

  return ifelse(isnan(score), 10000000, -score)

end