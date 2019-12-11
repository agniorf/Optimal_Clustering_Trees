function run_gmm(X, K; seed = 1, km_it_cnt = 20, em_it_cnt = 20, normalize = false)
	## Generally > 10 sufficient for km_it_cnt, em_it_cnt
	@rput X 
	@rput K
	@rput km_it_cnt
	@rput em_it_cnt
	@rput seed
	@rput normalize
	R"
	library(ClusterR)
	if (normalize){
		X = center_scale(X, mean_center = T, sd_scale = T)  
	}
	gmm = GMM(X, K, dist_mode = 'maha_dist', seed_mode = 'random_subset',
	  km_iter = km_it_cnt, em_iter = em_it_cnt, verbose = F, seed = seed)          
	## predict centroids, covariance matrix and weights
	pr = predict_GMM(X, gmm$centroids, gmm$covariance_matrices, gmm$weights)
	assignments <- pr$cluster_labels
	"
	@rget assignments
	return Array{Int64}(assignments)
end

function run_kmeansplus(X, K; seed = 1, init_cnt = 100, max_it_cnt = 100, normalize = false)
	@rput X 
	@rput K
	@rput init_cnt
	@rput max_it_cnt
	@rput seed
	@rput normalize
	R"
	library(ClusterR)
	if (normalize){
		X = center_scale(X, mean_center = T, sd_scale = T)  
	}
	km = KMeans_rcpp(X, clusters = K, 
	                 num_init = init_cnt, max_iters = max_it_cnt, 
	                 initializer = 'kmeans++', seed = seed)
	assignments <- km$clusters
	"
	@rget assignments
	return Array{Int64}(assignments)
end

function run_dbscan(X, epsilon; seed = 1, minpts = 5, normalize = false)
	@rput X 
	@rput epsilon
	@rput minpts
	@rput seed
	@rput normalize
	R"
	library(dbscan)

	if (normalize){
		X = center_scale(X, mean_center = T, sd_scale = T)  
	}

	getmode <- function(v) {
		uniqv <- setdiff(unique(v),0)
		t <- uniqv[which.max(tabulate(match(v, uniqv)))]
	}

	set.seed(seed)
	dbs <- dbscan(X, epsilon, minPts = minpts, weights = NULL, borderPoints = TRUE)
	assignments <- dbs$cluster
	neighbors <- kNN(X, minpts)
	ind_list = which(assignments == 0)
	for (i in(ind_list)){
	assignments[i] <- getmode(assignments[neighbors$id[i,]])
	}

	ind <- is.na(assignments)
	assignments[ind] <- max(c(assignments,0),na.rm=TRUE) + 1:sum(ind)
	"
	@rget assignments
	return Array{Int64}(assignments)
end

function run_hclust(X, K; seed = 1, m = "average", normalize = false)
	distance_matrix = create_distance_matrix_numeric(convert(Matrix{Float64},X))
	@rput distance_matrix
	@rput K
	@rput seed
	@rput m
	@rput normalize
	R"
	if (normalize){
		X = center_scale(X, mean_center = T, sd_scale = T)  
	}

	set.seed(seed)
	clusters <- hclust(as.dist(distance_matrix), method = m)
	assignments <- cutree(clusters, K)
	"
	@rget assignments
	return Array{Int64}(assignments)
end

function eval_method(X, param_range, seed, cr, method; normalize = false)
	@assert method in ["gmm", "kmeans_plus", "dbscan", "hclust"]
	score_dict = Dict{Float64,Float64}()
	assignments_dict = Dict{Float64,Array{Int64}}()
	X_t =  (convert(Matrix{Float64},X))'
	distance_matrix = create_distance_matrix_numeric(convert(Matrix{Float64},X))
	for k in param_range
		Random.seed!(seed)
		if method == "gmm"
			assignments = run_gmm(X, k, seed = seed, normalize = normalize)
		elseif method == "kmeans_plus"
			assignments = run_kmeansplus(X, k, seed = seed, normalize = normalize)
		elseif method == "dbscan"
			assignments = run_dbscan(X, k, seed = seed, normalize = normalize)
		elseif method == "hclust"
			assignments = run_hclust(X, k, seed = seed, normalize = normalize)
		end
		score_dict[k] = cluster_score(distance_matrix, assignments, cr)
		assignments_dict[k] = assignments
	end
	bestk = collect(keys(score_dict))[findmax(collect(values(score_dict)))[2]]
	println("Best K = $bestk: Score = $(round(score_dict[bestk],digits=3))")
	return bestk, assignments_dict[bestk]
end