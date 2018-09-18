using DataFrames, MLDataUtils
using Clustering, Distances
using RDatasets
using OptimalTrees


data = dataset("cluster", "ruspini");
dataset_array = convert(Array{Float64}, data);
dataset_t = dataset_array';

K = 3; srand(1234); 

kmeans_result = kmeans(dataset_t, K);
assignments = kmeans_result.assignments;
dataset_full = DataFrame(hcat(data, assignments));

X = dataset_full[1:end-1]; y = dataset_full[end];

distance_matrix = Distances.pairwise(Euclidean(), dataset_t);

function make_assignments(X, dim, threshold)
  col = X[:,dim]
  assignments = ifelse.(col.<= threshold, 1, 2)
  return Array(assignments)
end


function raw_error(assignments::Array{Int64,1}, distance_matrix::Array{Float64,2})
  # Bring in distances
  K = length(unique(assignments))

  distances = deepcopy(distance_matrix);
  assign_df = DataFrame(hcat(collect(1:size(assignments,1)), assignments));

  K = size(unique(assignments))
  n = size(assign_df,1)
  pairwise_n = n*(n-1)/2

  M = sum(distances)/(n*(n-1))
  
  intra_means = []
  clust_dict = Dict{Int64, Int64}()

  for clust in unique(assignments)

    println("Cluster number = ", clust)

    index_list = assign_df[assign_df[:x2] .== clust, :x1];
    clust_dict[clust] = index_list;
    cluster_distances = distance_matrix[index_list, index_list];

    clust_size = size(index_list,1)
    clust_pairs = size(index_list,1)*(size(index_list,1)-1) # not unique

    m_k = sum(cluster_distances)/clust_pairs
    push!(intra_means, m_k)

    f_k = (sum(cluster_distances .<= m_k) - clust_size)/clust_pairs

    clust_dict[clust] = f_k

  end

  pairs_dict = Dict{Tuple{Int64,Int64}, Int64}()

  for k in unique(assignments), l in setdiff(unique(assignments),k)
    if k < l
      cluster_distances = distance_matrix[clust_dict[k], clust_dict[l]]

      kl_pairs = length(cluster_distances)
      m_kl = sum(cluster_distances)/kl_pairs
      f_kl = sum(cluster_distances .<= m_kl)/kl_pairs

      pairs_dict[(k,l)] = f_kl
    end
  end

  scat = (1/K)*sum(clust_dict)/M 
  dens_bw = 1/(K*(K-1))*sum(sum(f_kl))

  println("\nScore = $score")
  return ifelse(isnan(score), 10000000, score)

end


#### Evaluate Splits

# for i = 1:100
#   assign = Array(ifelse.(X[2].<= 91, ifelse.(X[1].<= i,1,2),3))
#   score = raw_error(assign, distance_matrix)
#   println("Threshold = $(i): Raw error = $(score)")
# end

# assign_2clust = Array(ifelse.(X[2].<= 91,1,2));
# assign_3clust = Array(ifelse.(X[2].<= 91, ifelse.(X[1].<= 50,1,2),3));
assign_4clust = Array(ifelse.(X[2].<= 91, ifelse.(X[1].<= 50,1,2),ifelse.(X[1].<68.5,3,4)));
assign_5clust = Array(ifelse.(X[1].< 56.5, 
  ifelse.(X[2].< 106,1,2),
  ifelse.(X[1].<84,
    ifelse.(X[2].< 62.5,3,4),5)));

assign_5clust = Array(ifelse.(X[1].<84,ifelse.(X[2].<= 91, ifelse.(X[1].<= 50,1,2), ifelse.(X[1].<68.5,3,4)),5));


# raw_error(assign_2clust, distance_matrix);
# raw_error(assign_3clust, distance_matrix);
raw_error(assign_4clust, distance_matrix);
raw_error(assign_5clust, distance_matrix);