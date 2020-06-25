library(data.table)
library(tidyverse)
library(stringr)
library(gridExtra)

setwd("../results_kmeans0/")

filenames <- list.files(pattern="geom0.99-ws_oct.csv$", full.names=TRUE)
filenames_assign <- list.files(pattern="geom0.99-ws_oct_assignments.csv$", full.names=TRUE)
df <- data.frame()

for (filename in filenames) {
  file_short <- substr(filename,3,nchar(filename)-4);
  
  ## Add ICOT/OCT results
  df_next <- read.table(file = filename, sep = ",", header = T) %>% 
    filter(method %in% c("ICOT_local", "OCT"))
  df_next$filename <- file_short
  df <- rbind(df, df_next)
  
  ## Add R results
  df_next <- read.table(file = paste0("../results_R/",filename), sep = ",", header = T)
  df_next$filename <- file_short
  df <- rbind(df, df_next)
} 

df_assign <- data.frame()
for (filename in filenames_assign) {
  file_short <- substr(filename,3,nchar(filename)-4);
  df_next <- read.table(file = paste0("../results_R/",filename), sep = ",", header = T) %>%
    summarize_each(n_distinct)
  df_next$filename <- file_short
  df_assign <- rbind(df_assign, df_next)
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
  select(data, result_cnt, ICOT_local, OCT, kmeans_plus, hclust, gmm, dbscan,True)

### Dunn Table
df %>% filter(criterion == "dunnindex") %>%
  select(data, method, dunn) %>%
  group_by(data, method) %>%
  summarize(result_cnt = n(), 
            metric_score = mean(dunn)) %>%
  spread(method, metric_score) %>%
  select(data, result_cnt, ICOT_local, OCT, kmeans_plus, hclust, gmm, dbscan,True)


### Runtime
df %>% filter(criterion == "dunnindex") %>%
  select(data, method, runtime) %>%
  group_by(data, method) %>%
  summarize(result_cnt = n(), 
            runtime = mean(runtime)/60) %>%
  spread(method, runtime) %>%
  select(data, result_cnt, ICOT_local, OCT, kmeans_plus, hclust, gmm, dbscan)

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

### Sensitivity to training criterion choice
df %>%
  group_by(criterion) %>%
  summarize(sm = mean(silhouette),
            di = mean(dunn)) %>%
  arrange(desc(criterion))

(.149-.177)/.177
(.416-.475)/.475


# Compare assignments (6/25/2020) -----------------------------------------

library(aricode)
data <- c("Atom", "Chainlink", "EngyTime",
          "Hepta", "Lsun", "Target",
          "Tetra", "TwoDiamonds", "WingNut")
df_nmi = head(assignments,0)
df_nmi$data

res = do.call(rbind, lapply(data, function(i){
  assignments = read.table(paste0('./',i,'-silhouette-seed5-geom0.99-ws_oct_assignments.csv'), sep = ",", header = T)
  res = sapply(names(df_nmi), function(col){NMI(assignments$truth, pull(assignments[col]), variant = 'joint')})
  res = as.data.frame(t(res))
  res$name = i
  return(res)
}))

res %>% summary()

                      