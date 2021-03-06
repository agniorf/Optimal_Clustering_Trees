using DataFrames, MLDataUtils
using Clustering, Distances
using RDatasets
# using OptimalTrees

######## LSUN
dataset_full = readtable("../Experiments/Experiment_1/data/TwoDiamonds.csv"); 
# dataset = readtable("../Testing_Class_Project/data/TwoDiamonds.csv")
dataset_t = convert(Array{Float64}, dataset_full[1:end-1])';
assign_3clust = Array(dataset_full[end]);
assign_4clust = Array(ifelse.(X[2].<= 1.75, ifelse.(X[1].<= 2.1,1,2),ifelse.(X[1].<1.7,3,4)));



######## RUSPINI 
# data = dataset("cluster", "ruspini");
# dataset_array = convert(Array{Float64}, data);
# dataset_t = dataset_array';

# K = 3; srand(1234); kmeans_result = kmeans(dataset_t, K); assignments = kmeans_result.assignments;
# dataset_full = DataFrame(hcat(data, assignments));

# assign_1clust = convert(Array{Int64,1},ones(75));
# assign_2clust = Array(ifelse.(X[2].<= 91,1,2));
# assign_3clust = Array(ifelse.(X[2].<= 91, ifelse.(X[1].<= 50,1,2),3));
# assign_4clust = Array(ifelse.(X[2].<= 91, ifelse.(X[1].<= 50,1,2),ifelse.(X[1].<68.5,3,4)));
# # assign_5clust = Array(ifelse.(X[1].< 56.5, 
# #   ifelse.(X[2].< 106,1,2),
# #   ifelse.(X[1].<84,
# #     ifelse.(X[2].< 62.5,3,4),5)));
# assign_5clust = Array(ifelse.(X[1].<84,ifelse.(X[2].<= 91, ifelse.(X[1].<= 50,1,2), ifelse.(X[1].<68.5,3,4)),5));

# assign_50 = Array(ifelse.(X[2].<= 50,1,2));
# assign_91 = Array(ifelse.(X[2].<= 91,1,2));


X = dataset_full[1:end-1]; y = dataset_full[end];

distance_matrix = Distances.pairwise(Euclidean(), dataset_t);


### Process data

function make_assignments(X, dim, threshold)
  col = X[:,dim]
  assignments = ifelse.(col.<= threshold, 1, 2)
  return Array(assignments)
end


function raw_error(assignments::Array{Int64,1}, distance_matrix::Array{Float64,2})
  # Bring in distances
  distances = deepcopy(distance_matrix);
  assign_df = DataFrame(hcat(collect(1:size(assignments,1)), assignments));

  K = length(unique(assignments))
  n = size(assign_df,1)
  pairwise_n = n*(n-1)/2

  M = sum(distances)/(n*(n-1))
  
  intra_means = []
  intra_n = 0
  clust_dict = Dict{Int64, Array{Int64,1}}()
  intra_f_dict = Dict{Int64, Float64}()

  for clust in unique(assignments)
    index_list = assign_df[assign_df[:x2] .== clust, :x1];
    clust_dict[clust] = index_list;
    cluster_distances = distance_matrix[index_list, index_list];

    clust_size = size(index_list,1)
    clust_pairs = size(index_list,1)*(size(index_list,1)-1) # not unique

    m_k = sum(cluster_distances)/clust_pairs

    intra_n += clust_pairs
    push!(intra_means, sum(cluster_distances))

    f_k = (sum(cluster_distances .< m_k) - clust_size)/clust_pairs
    # println("K = $clust (size = $(clust_size)): m_k = $(m_k), f_k = $(f_k)")

    intra_f_dict[clust] = f_k

  end

  pairs_f_dict = Dict{Tuple{Int64,Int64}, Float64}()

  between_n = 0
  for k in unique(assignments), l in setdiff(unique(assignments),k)
    if k < l
      cluster_distances = distance_matrix[clust_dict[k], clust_dict[l]]

      kl_pairs = length(cluster_distances)
      between_n += kl_pairs

      # m_kl = sum(cluster_distances)/kl_pairs
      m_kl = minimum(cluster_distances)
      
      # f_kl = sum(cluster_distances .<= m_kl)/kl_pairs
      std_d = std(cluster_distances)
      # f_kl = sum(cluster_distances .< (m_kl -std_d))
      f_kl = sum(cluster_distances .< (m_kl +std_d))
      println("(k,l) = ($k,$l): m_kl = $(m_kl), f_kl = $(f_kl), std = $(std_d)")
      # println("(k,l) = ($k,$l): m_kl = $(m_kl), f_kl = $(f_kl)")
      pairs_f_dict[(k,l)] = f_kl

    end
  end

  scat = sum(intra_means)/((intra_n)*M)
  dens_bw_inner = 0
  for l=1:K, k=1:(l-1)
    dens_bw_inner += pairs_f_dict[(k,l)]
    # dens_bw_inner += pairs_f_dict[(k,l)]/max(intra_f_dict[k], intra_f_dict[l])
  end

  # dens_bw = 2/(K*(K-1))*dens_bw_inner
  dens_bw = dens_bw_inner/between_n

  score = (scat + dens_bw)/2
  # println("Density: ", dens_bw, "; Scatter: ", scat)
  # println("Score = $score")
  return ifelse(isnan(score), 1000, score)

end


#### Evaluate Splits

for i = .1:.1:5
  assign = Array(ifelse.(X[2].<= i,1,2))
  score = raw_error(assign, distance_matrix)
  println("Threshold = $(i): Raw error = $(score)")
end

assign_1clust = convert(Array{Int64,1},ones(75));
assign_2clust = Array(ifelse.(X[2].<= 91,1,2));
assign_3clust = Array(ifelse.(X[2].<= 91, ifelse.(X[1].<= 50,1,2),3));
assign_4clust = Array(ifelse.(X[2].<= 91, ifelse.(X[1].<= 50,1,2),ifelse.(X[1].<68.5,3,4)));
# assign_5clust = Array(ifelse.(X[1].< 56.5, 
#   ifelse.(X[2].< 106,1,2),
#   ifelse.(X[1].<84,
#     ifelse.(X[2].< 62.5,3,4),5)));
assign_5clust = Array(ifelse.(X[1].<84,ifelse.(X[2].<= 91, ifelse.(X[1].<= 50,1,2), ifelse.(X[1].<68.5,3,4)),5));
assign_50 = Array(ifelse.(X[2].<= 50,1,2));
assign_91 = Array(ifelse.(X[2].<= 91,1,2));

println("2 clust: ", raw_error(assign_2clust, distance_matrix));
println("3 clust: ", raw_error(assign_3clust, distance_matrix));
println("4 clust: ", raw_error(assign_4clust, distance_matrix));
println("5 clust: ", raw_error(assign_5clust, distance_matrix));

s_dbw(X, assign_2clust)
s_dbw(X, assign_3clust)
s_dbw(X, assign_4clust)
s_dbw(X, assign_5clust)


