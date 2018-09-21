using DataFrames, MLDataUtils
using Clustering, Distances
using RDatasets
using OptimalTrees
using StatsBase


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

#### Evaluate Splits

for i = 1:160
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

function raw_error(assignments::Array{Int64,1}, distance_matrix::Array{Float64,2})
  # Bring in distances
  K = length(unique(assignments))

  distances = deepcopy(distance_matrix);
  assign_df = DataFrame(hcat(collect(1:size(assignments,1)), assignments));

  
  K = length(unique(assignments))
  n = size(assign_df,1)
  pairwise_n = n*(n-1)/2

  M = sum(distances)/(n*(n-1))
  
  intra_means = []
  intra_n = 0
  min_f_dict = Dict{Int64, Float64}()
  max_f_dict = Dict{Int64, Float64}()

  s_score = 0
  for clust in unique(assignments)
    index_list = assign_df[assign_df[:x2] .== clust, :x1];
    n_k = size(index_list,1);
    cluster_distances = distance_matrix[index_list, index_list];
    max_k = maximum(cluster_distances);
    # max_f_dict[clust] = m_k;

    not_clust_ind = setdiff(collect(1:n),index_list);
    inter_cluster_distances = distance_matrix[index_list, not_clust_ind];
    min_k = minimum(inter_cluster_distances);
    # min_f_dict[clust] = min_k;
   
    s_score+= n_k*(min_k/max_k) ;
  end


  score = s_score/n;
  # println("Density: ", dens_bw, "; Scatter: ", scat)
  # println("Score = $score")
  return ifelse(isnan(score), 1000, -score)

end

println("2 clust: ", raw_error(assign_2clust, distance_matrix));
println("3 clust: ", raw_error(assign_3clust, distance_matrix));
println("4 clust: ", raw_error(assign_4clust, distance_matrix));
println("5 clust: ", raw_error(assign_5clust, distance_matrix));


