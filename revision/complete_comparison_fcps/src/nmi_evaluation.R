library(aricode)
library(gridExtra)
library(foreign)
library(tidyverse)

setwd('../results')
datasets <- c("Atom", "Chainlink", "EngyTime",
          "Hepta", "Lsun", "Target",
          "Tetra", "TwoDiamonds", "WingNut")

clus_assignments <- function(file, metric, s){
  clus_params = read.csv("../../clus_comparison/clus_results_noprune.csv")
  depth = clus_params %>% filter(data == file & criterion == metric & seed == s) %>% pull(K)
  result_file <- read.arff(paste0("../../clus_comparison/experiments_noprune/",file,"_depth",depth,"_seed",s,".train.1.pred.arff"))
  assignments <- result_file$`Pruned-models` %>% as.integer()
  return(assignments)
}


iter_list = expand.grid(datasets, 1:5, c('silhouette','dunnindex'))

result_byseed = do.call(rbind, mapply(function(i,seed,metric){
  print(paste0("Data = ",i, "; Seed = ", seed))
  assignments = read.table(paste0('./',i,'-',metric,'-seed',seed,'-geom0.99-ws_oct_assignments.csv'), 
                           sep = ",", header = T)
  clus = clus_assignments(i,metric,seed)
  assignments$clus = clus
  res = sapply(names(assignments), function(col){NMI(assignments$truth, pull(assignments[col]))})
  res = as.data.frame(t(res))
  res$name = i
  res$seed = seed
  res$metric = metric
  return(res)
}, iter_list[,1], iter_list[,2], iter_list[,3],SIMPLIFY = FALSE))

result <- result_byseed %>%
  group_by(name,metric) %>%
  summarize_at(vars(-'seed'), list(mean)) %>%
  mutate_if(is.numeric, ~round(.,digits = 3))

result %>% group_by(metric) %>% summarize_if(is.numeric, funs(mean(., na.rm = TRUE)))
