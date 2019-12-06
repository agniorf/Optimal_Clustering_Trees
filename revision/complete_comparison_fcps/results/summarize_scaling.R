library(data.table)
library(tidyverse)

setwd("../results/")

filenames <- list.files(pattern="ws_[a-z]*.csv$", full.names=TRUE)

df <- data.frame()

for (filename in filenames) {
  file_short <- substr(filename,3,nchar(filename)-4);
  
  df_next <- read.table(file = filename, sep = ",", header = T)
  df_next$filename <- file_short
  df <- rbind(df, df_next)
} 

# setwd("../results_fullyscaled/")
# final_cols <- names(df)[1:(ncol(df)-1)]
# filenames_scaled <- list.files(pattern="*seed\\d\\.csv$", full.names=TRUE)
# 
# for (filename in filenames_scaled) {
#   file_short <- substr(filename,3,nchar(filename)-4);
#   
#   df_raw <- read.table(file = filename, sep = ",", header = T)
#   df_raw$geom_threshold <- 0.99
#   df_raw$warm_start <- "oct"
#   df <- df_raw %>% select(final_cols)
#   write.csv(df, file = paste0("../results/",file_short, "-geom0.99-ws_oct.csv"), row.names = FALSE)
#   
#   df_assign <- read.table(file = paste0("../results_fullyscaled/", file_short, "_assignments.csv"), sep = ",", header = T)
#   write.csv(df_assign, file = paste0("../results/", file_short, "-geom0.99-ws_oct_assignments.csv"), row.names = FALSE)
# } 


### Check job completion
data <- c("Atom", "Chainlink", "EngyTime",
          "Hepta", "Lsun", "Target",
          "Tetra", "TwoDiamonds", "WingNut")
seeds <- c(1,2,3,4,5)
criterion <- c("silhouette","dunnindex")
thresholds = c(0.0,0.9,.99);
ws = c("none","oct")

df_match <- df %>% filter(method == "ICOT_local") %>%
  select("data","criterion","geom_threshold","warm_start","seed","runtime")

job_status <- as.data.frame(expand.grid(data, criterion, thresholds, ws, seeds)) %>%
  `colnames<-`(c("data","criterion","geom_threshold","warm_start","seed")) %>%
  left_join(., df_match, by = c("data", "criterion", "geom_threshold", "warm_start", "seed")) %>%
  arrange(data, criterion, geom_threshold, warm_start, seed)

write.csv(job_status, "../scaling_job_status.csv", row.names = FALSE)

write.csv(subset(job_status, is.na(runtime), data != "EngyTime"), "failed_parameters.csv", row.names = FALSE)

### Runtime
df_match %>% filter(criterion == "silhouette") %>% 
  group_by(geom_threshold, warm_start) %>%
  summarize(avg_runtime = mean(runtime/60)) 


### Silhouette Table
df %>% filter(criterion == "silhouette") %>%
  select(data, method, silhouette) %>%
  group_by(data, method) %>%
  summarize(result_cnt = n(), 
            metric_score = mean(silhouette)) %>%
  spread(method, metric_score) %>%
  select(data, result_cnt, ICOT_local, dbscan, gmm, hclust, kmeans_plus, OCT,True)

### Dunn Table
df %>% filter(criterion == "dunnindex") %>%
  select(data, method, dunn) %>%
  group_by(data, method) %>%
  summarize(result_cnt = n(), 
            metric_score = mean(dunn)) %>%
  spread(method, metric_score) %>%
  select(data, result_cnt, ICOT_local, dbscan, gmm, hclust, kmeans_plus, OCT,True)

