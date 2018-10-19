setwd("~/Packages/Optimal_Clustering_Trees/Experiments/Experiment_Real_World/Hubway")

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
clusters <- kmeans(df_norm, 7)
df$cluster<- clusters$cluster

#Number of points in each cluster
table(clusters$cluster)
df$cluster<- clusters$cluster
new<-as.data.frame(clusters$centers)
df_cent<-round(mapply(denormalize, df[,1:9], new))

k.max=10
wss <- sapply(1:k.max,function(k){kmeans(df_norm, k, nstart=50,iter.max = 15)$tot.withinss})
plot(1:k.max, wss,
     type="b", pch = 19, frame = FALSE, 
     xlab="Number of clusters K",
     ylab="Total within-clusters sum of squares")


#Sample multiple datasets for a given size

numSeeds <-10
numRows <-500
data <- df
k<-4

randomSampling<-function(data, numSeeds, numRows){
  for (i in 1:numSeeds) {
    destfile <- paste0("~/Packages/Optimal_Clustering_Trees/Experiments/Experiment_Real_World/Hubway/data/random/random_hubway_sample_seed",i, "_obs_",numRows,".csv")
    set.seed(i)    
    df_s<-data[sample(nrow(data), numRows), ]
    write.csv(df_s, destfile, row.names = FALSE)
  }}
#50 different seeds for 500 random observations
randomSampling(df[,1:9], 50, 500)
randomSampling(df[,1:9], 50, 1000)
randomSampling(df[,1:9], 50, 1500)
randomSampling(df[,1:9], 50, 2000)

stratifiedSampling<-function(df, numSeeds, numRows,k){
  df_norm<-as.data.frame(lapply(df, normalize)) 
  set.seed(20)
  clusters <- kmeans(df_norm, k)
  df$cluster<- clusters$cluster
  for (i in 1:numSeeds) {
    destfile <- paste0("~/Packages/Optimal_Clustering_Trees/Experiments/Experiment_Real_World/Hubway/data/stratified/strat_hubway_numofclust_",k,"_sample_seed_",i, "_obs_",numRows,".csv")
    set.seed(i)    
    df_s<-stratified(df, "cluster", numRows, replace = FALSE)
    write.csv(df_s, destfile, row.names = FALSE)
  }}

for (k in 1:10) {
  stratifiedSampling(df, 50, 500,k)
  stratifiedSampling(df, 50, 1000,k)
  stratifiedSampling(df, 50, 1500,k)
  stratifiedSampling(df, 50, 2000,k)
}


stratified(data, "cluster", 500, replace = FALSE)

