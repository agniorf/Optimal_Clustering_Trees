setwd("~/Packages/Optimal_Clustering_Trees/Experiments/Experiment_Real_World/Hubway")
setwd("~/git/Optimal_Clustering_Trees/Experiments/Experiment_Real_World/Hubway_LS_mod")

df<- read.csv("HubwayTrips.csv")
#We read in from the Analytics Edge the clusters and we sample from there the data.
normalize <- function(temp) {
  temp <- (temp - mean(temp)) / sd(temp) 
  return(temp)
}
denormalize <- function(orig, new) {
  temp <- (new * sd(orig)) + mean(orig)
  return(temp)
}

df_norm<-as.data.frame(lapply(df, normalize)) 
set.seed(20)
clusters <- kmeans(df_norm, 5)
df$cluster<- clusters$cluster

#Number of points in each cluster
table(clusters$cluster)
df$cluster<- clusters$cluster
new<-as.data.frame(clusters$centers)
df_cent<-round(mapply(denormalize, df[,1:9], new))

#Sample multiple datasets for a given size

numSeeds <-1
numRows <-500
data <- df
k<-4

randomSampling<-function(data, numSeeds, numRows){
  for (i in 1:numSeeds) {
    destfile <- paste0("data/random_hubway_sample_seed",i, "_obs_",numRows,".csv")
    set.seed(i)    
    df_s<-data[sample(nrow(data), numRows), ]
    write.csv(df_s, destfile, row.names = FALSE)
  }}
#50 different seeds for 500 random observations
randomSampling(df[,1:9], 1, 2000)
randomSampling(df[,1:9], 1, 5000)
randomSampling(df[,1:9], 1, 10000)
randomSampling(df[,1:9], 1, 20000)
randomSampling(df[,1:9], 1, 50000)
randomSampling(df[,1:9], 1, 100000)

# stratifiedSampling<-function(df, numSeeds, numRows,k){
#   df_norm<-as.data.frame(lapply(df, normalize)) 
#   set.seed(20)
#   clusters <- kmeans(df_norm, k)
#   df$cluster<- clusters$cluster
#   for (i in 1:numSeeds) {
#     destfile <- paste0("data/stratified/strat_hubway_numofclust_",k,"_sample_seed_",i, "_obs_",numRows,".csv")
#     set.seed(i)    
#     df_s<-stratified(df, "cluster", numRows, replace = FALSE)
#     write.csv(df_s, destfile, row.names = FALSE)
#   }}

# for (k in 1:10) {
#   stratifiedSampling(df, 50, 500,k)
#   stratifiedSampling(df, 50, 1000,k)
#   stratifiedSampling(df, 50, 1500,k)
#   stratifiedSampling(df, 50, 2000,k)
# }
# 
# 
# stratified(data, "cluster", 500, replace = FALSE)
# 
