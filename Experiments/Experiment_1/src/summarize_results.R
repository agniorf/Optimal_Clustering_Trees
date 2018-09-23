library(data.table)
library(tidyverse)

# setwd("~/Packages/Optimal_Clustering_Trees/Experiments/Experiment_1/results")

setwd("~/git/Optimal_Clustering_Trees/Experiments/Experiment_1/results/newdunn_min10")

filenames <- list.files(pattern="*.csv$", full.names=TRUE)
# filenames <- setdiff(filenames, "./summary_withrobustdunn05minbucket20.csv")
df <- data.frame()
df_params <- data.frame()

for (filename in filenames) {
  str = filename
  x <- unlist(strsplit(str, "-"))
  dataname <- substr(x[2],1,nchar(x[2])-4); metric <- x[3]; method <- substr(x[4],1,nchar(x[4])-4);
  
  df_next <- read.table(file = filename, sep = ",", header = T)
  df_next$dataname <- dataname
  df_next$metric <- metric
  df_next$method <- method
  df <- rbind(df, df_next)
  
  df_next <- read.table(file = filename, sep = ",", header = T)
  df_params <- rbind(df_params, df_next)
}

write.csv(df, "summary_newdunn_min10.csv", row.names = F)


# temp = list.files(pattern="*.csv")
# temp_csv<-temp[which(!temp %like% "jld")]
# 
# myfiles = lapply(temp_csv, read.delim)
# 
# datasetlist <- c("Hepta","Lsun","Target",
#                     "Tetra","TwoDiamonds","Atom","WingNut") 
# metriclist <- c("silhouette","dunnindex","robustdunn")
# methodlist <- c("greedy","localsearch")
# 
# df <- data.frame()
# df_params <- data.frame()
# 
# for (metric in metriclist) {
#   for (dataname in datasetlist) {
#     for (method in methodlist) {
#     filename <- paste0("results-",
#                        dataname,".csv-",metric,"-",method,".csv")
#     df_next <- read.table(file = filename, sep = ",", header = T)
#     df_next$dataname <- dataname
#     df_next$metric <- metric
#     df_next$method <- method
#     df <- rbind(df, df_next)
#     filename <- paste0("results-",
#                        dataname,".csv-",metric,"-",method,".csv")
#     df_next <- read.table(file = filename, sep = ",", header = T)
#     df_params <- rbind(df_params, df_next)
#     }
#   }
# }
# 
# write.csv(df, "summary_withrobustdunn.csv", row.names = F)
# 
# df[which(df$score_optclust>=df$score_kmeans_bestk),]