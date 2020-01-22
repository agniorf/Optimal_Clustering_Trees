# install.packages("foreign")
library(foreign)
library(tidyverse)
library(ClusterR)
for (file in c("Atom", "Chainlink", "EngyTime",
              "Hepta", "Lsun", "Target",
              "Tetra", "TwoDiamonds", "WingNut")) {
  for (depth in c(1,2,3)){
    for (seed in 1:5){
      ## For a given dataset, read in the csv.
      df <- read.csv(paste0("data/", file, ".csv"), stringsAsFactors = FALSE)
      ## Delete label and save modified file in ARFF format
      df$label <- NULL
      n = nrow(df)
      p = ncol(df)
      df_normalized = center_scale(df, mean_center = T, sd_scale = T) %>% as.data.frame() %>%
        `colnames<-`(names(df)[1:p])
      data_file <- write.arff(df_normalized, paste0("experiments/",file,"_depth",depth,"_seed",seed,".arff"))
    
      
      ## Set up experiments
      experiment_script <- paste0("[Data]
                                  PruneSet = 0.25
                                  TestSet = None
                                  
                                  [General]
                                  RandomSeed = ",seed,"

                                  [Constraints]
                                  MaxDepth = ",depth,"
                                  
                                  [Attributes]
                                  Target = 1-",p, "
                                  Clustering = 1-",p, " 
                                  Descriptive = 1-",p, " 
                                  
                                  [Tree]
                                  Heuristic = VarianceReduction
                                  
                                  [Output]
                                  WritePredictions = {Train}
                                  ")
      write_file(experiment_script, paste0("experiments/",file,"_depth",depth,"_seed",seed,".s"))
    }
  }
}

# c("Atom", "Chainlink", "EngyTime", "Hepta", "Lsun", "Target", "Tetra", "TwoDiamonds", "WingNut")

## Run commands
for (file in c("Atom", "Chainlink", "EngyTime",
               "Hepta", "Lsun", "Target",
               "Tetra", "TwoDiamonds", "WingNut")) {
  for (depth in c(1,2,3)){
    for (seed in 1:5){
      cat("\njava -jar ~/software/Clus/Clus.jar experiments/",file,"_depth",depth,"_seed",seed,".s", 
          sep = "")
}}}
## Look at results
result_file <- read.arff(paste0(file,"_depth",depth,"_seed",seed,".train.1.pred.arff"))
X <- result_file[,1:p]
assignments <- result_file$`Pruned-models` %>% as.integer()