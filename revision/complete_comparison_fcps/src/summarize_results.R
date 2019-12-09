library(data.table)
library(tidyverse)

setwd("../results_kmeans0/")

filenames <- list.files(pattern="geom0.99-ws_oct.csv$", full.names=TRUE)
# filenames_assign <- list.files(pattern="assignments.csv$", full.names=TRUE)
# filenames <- list.files(pattern="seed\\d\\.csv$", full.names=TRUE)

df <- data.frame()

for (filename in filenames) {
  file_short <- substr(filename,3,nchar(filename)-4);
  
  df_next <- read.table(file = filename, sep = ",", header = T)
  df_next$filename <- file_short
  df <- rbind(df, df_next)
} 

#write.csv(df, "../results_summary_dec3_noscaling.csv", row.names = F)

### Do results vary across seeds?
df %>% group_by(data, criterion, method) %>%
  summarize(result_cnt = n(),
            unique_sil = n_distinct(silhouette),
            unique_dunn = n_distinct(dunn)) %>%
  filter(unique_sil + unique_dunn > 2)

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


### Runtime
df %>% filter(criterion == "dunnindex") %>%
  select(data, method, runtime) %>%
  group_by(data, method) %>%
  summarize(result_cnt = n(), 
            runtime = mean(runtime)/60) %>%
  spread(method, runtime)

### Check job completion
data <- c("Atom", "Chainlink", "EngyTime",
          "Hepta", "Lsun", "Target",
          "Tetra", "TwoDiamonds", "WingNut")
seeds <- c(1,2,3,4,5)
criterion <- c("silhouette","dunnindex")

df_match <- df %>% filter(method == "ICOT_local") %>%
  select("data","criterion","seed","runtime")

job_status <- as.data.frame(expand.grid(data, criterion, seeds)) %>%
  `colnames<-`(c("data","criterion","seed")) %>%
  left_join(., df_match) %>%
  mutate(index = c(1:90))
