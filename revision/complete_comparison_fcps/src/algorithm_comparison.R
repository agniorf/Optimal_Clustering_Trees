library(ClusterR)
library(dbscan)

## 
df <- read.csv("../../data/FCPS/01FCPSdata/Lsun.csv")
X = df[, -ncol(df)]   # data (excluding the response variable)
y = df[, ncol(df)]    # the response variable
dat = center_scale(X, mean_center = T, sd_scale = T)  # centering and scaling the data

############ GMM Model ################
## CV and train on K with lowest BIC
# Key parameters: clusters (K), km_iter, em_iter, dist_mode
opt_gmm = Optimal_Clusters_GMM(dat, max_clusters = 10, criterion = "BIC", 
                               dist_mode = "maha_dist", seed_mode = "random_subset",
                               km_iter = 10, em_iter = 10, var_floor = 1e-10, 
                               plot_data = T)

gmm = GMM(dat, which.min(opt_gmm), dist_mode = "maha_dist", seed_mode = "random_subset",
          km_iter = 10, em_iter = 10, verbose = F)          

## predict centroids, covariance matrix and weights
pr = predict_GMM(dat, gmm$centroids, gmm$covariance_matrices, gmm$weights)
gmm_assignments <- pr$cluster_labels

############ K-MEANS++ ##############
# Key parameters: clusters (K), num_init, max_iters
opt_km <- Optimal_Clusters_KMeans(dat, max_clusters = 10, criterion = "distortion_fK",
                                  plot_clusters = TRUE)
km = KMeans_rcpp(dat, clusters = which.min(opt_km), 
                 num_init = 5, max_iters = 100, initializer = 'kmeans++')
kmplus_assignments <- km$clusters

########### DBSCAN ##############
# Key parameters: eps (higher = fewer clusters), minPts
kNNdistplot(X, k = 5)
eps <- .3
dbs <- dbscan(X, eps, minPts = 5, weights = NULL, borderPoints = TRUE)
dbs_assignments <- dbs$cluster
### Assign all outliers to their own cluster
ind <- dbs_assignments == 0
dbs_assignments[ind] <- max(dbs_assignments) + 1:sum(ind)

plot(dat, col=dbs$cluster)
points(dat[dbs$cluster==0,], pch = 3, col = "grey")
hullplot(dat, dbs)


########## HIERARCHICAL #########
clusters <- hclust(dist(X), method = 'average')
