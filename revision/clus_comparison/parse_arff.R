# install.packages("foreign")
library(foreign)
library(tidyverse)

file <- "Lsun"

for (depth in c(1,2,3)){
  ## For a given dataset, read in the csv.
  df <- read.csv(paste0("data/", file, ".csv"), stringsAsFactors = FALSE)
  ## Delete label and save modified file in ARFF format
  df$label <- NULL
  data_file <- write.arff(df, paste0(file,"/depth",depth,".arff"))
  n = nrow(df)
  p = ncol(df)
  
  ## Set up experiments
  experiment_script <- paste0("[Data]
PruneSet = 0.25
TestSet = None

[General]
RandomSeed = 1

[Model]
MinimalWeight = 5.0

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
  write_file(experiment_script, paste0(file,"/depth",depth,".s"))
  
}



## Look at results
result_file <- read.arff(paste0(file,"Lsun.train.1.pred.arff"))
assignments <- result_file$`Pruned-models` %>% as.integer()
