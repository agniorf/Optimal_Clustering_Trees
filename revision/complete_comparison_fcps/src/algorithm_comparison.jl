function run_gmm(X, K; km_it_cnt = 10, em_it_cnt = 10)
	@rput X 
	@rput K
	@rput km_it_cnt
	@rput em_it_cnt
	R"
	library(ClusterR)
	gmm = GMM(X, K, dist_mode = 'maha_dist', seed_mode = 'random_subset',
	  km_iter = km_it_cnt, em_iter = em_it_cnt, verbose = F)          
	## predict centroids, covariance matrix and weights
	pr = predict_GMM(X, gmm$centroids, gmm$covariance_matrices, gmm$weights)
	assignments <- pr$cluster_labels
	"
	@rget assignments
	return Array{Int64}(assignments)
end

function run_kmeansplus(X, K; init_cnt = 5, max_it_cnt = 100)
	@rput X 
	@rput K
	@rput init_cnt
	@rput max_it_cnt
	R"
	library(ClusterR)
	km = KMeans_rcpp(X, clusters = K, 
	                 num_init = init_cnt, max_iters = max_it_cnt, initializer = 'kmeans++')
	assignments <- km$clusters
	"
	@rget assignments
	return Array{Int64}(assignments)
end

function run_dbscan(X, epsilon; minpts = 5)
	@rput X 
	@rput epsilon
	@rput minpts
	R"
	library(dbscan)
	dbs <- dbscan(X, epsilon, minPts = minpts, weights = NULL, borderPoints = TRUE)
	assignments <- dbs$cluster
	ind <- assignments == 0
	assignments[ind] <- max(assignments) + 1:sum(ind)
	"
	@rget assignments
	return Array{Int64}(assignments)
end

function run_hclust(X, K)
	distance_matrix = create_distance_matrix_numeric(convert(Matrix{Float64},X))
	@rput distance_matrix
	@rput K
	R"
	clusters <- hclust(as.dist(distance_matrix), method = 'average')
	assignments <- cutree(clusters, K)
	"
	@rget assignments
	return Array{Int64}(assignments)
end

function eval_method(X, param_range, seed, cr, method)
	@assert method in ["gmm", "kmeans_plus", "dbscan", "hclust"]
	score_dict = Dict{Float64,Float64}()
	assignments_dict = Dict{Float64,Array{Int64}}()
	X_t =  (convert(Matrix{Float64},X))'
	distance_matrix = create_distance_matrix_numeric(convert(Matrix{Float64},X))
	for k in param_range
		Random.seed!(seed)
		if method == "gmm"
			assignments = run_gmm(X, k)
		elseif method == "kmeans_plus"
			assignments = run_kmeansplus(X, k)
		elseif method == "dbscan"
			assignments = run_dbscan(X, k)
		elseif method == "hclust"
			assignments = run_hclust(X, k)
		end
		score_dict[k] = cluster_score(distance_matrix, assignments, cr)
		assignments_dict[k] = assignments
	end
	bestk = collect(keys(score_dict))[findmax(collect(values(score_dict)))[2]]
	println("Best K = $bestk: Score = $(round(score_dict[bestk],digits=3))")
	return bestk, assignments_dict[bestk]
end