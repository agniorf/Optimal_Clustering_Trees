library(data.table)
library(tidyverse)

setwd("~/Packages/Optimal_Clustering_Trees/Experiments/Experiment_1/results")
# setwd("~/git/Optimal_Clustering_Trees/Experiments/Experiment_1/results")
temp = list.files(pattern="*.csv")
temp_csv<-temp[which(!temp %like% "jld")]

myfiles = lapply(temp_csv, read.delim)

datasetlist <- c("Hepta","Lsun","Target",
                    "Tetra","TwoDiamonds","Atom","WingNut") 
metriclist <- c("silhouette","dunnindex","robustdunn")
methodlist <- c("greedy","localsearch")

df <- data.frame()
df_params <- data.frame()

for (metric in metriclist) {
  for (dataname in datasetlist) {
    for (method in methodlist) {
    filename <- paste0("results-",
                       dataname,".csv-",metric,"-",method,".csv")
    df_next <- read.table(file = filename, sep = ",", header = T)
    df_next$dataname <- dataname
    df_next$metric <- metric
    df_next$method <- method
    df <- rbind(df, df_next)
    filename <- paste0("results-",
                       dataname,".csv-",metric,"-",method,".csv")
    df_next <- read.table(file = filename, sep = ",", header = T)
    df_params <- rbind(df_params, df_next)
    }
  }
}

write.csv(df, "summary.csv", row.names = F)

df[which(df$score_optclust>=df$score_kmeans_bestk),]



