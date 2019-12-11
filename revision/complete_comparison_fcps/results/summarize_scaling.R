library(data.table)
library(tidyverse)
library("RColorBrewer")


setwd("../results_kmeans0/")

filenames <- list.files(pattern="ws_[a-z]*.csv$", full.names=TRUE)

df <- data.frame()

for (filename in filenames) {
  file_short <- substr(filename,3,nchar(filename)-4);
  
  df_next <- read.table(file = filename, sep = ",", header = T)
  df_next$filename <- file_short
  df <- rbind(df, df_next)
} 

write.csv(df, file = paste0("../full_scaling_results_kmeans0.csv"), row.names = FALSE)

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

# 
job_status <- as.data.frame(expand.grid(data, criterion, thresholds, ws, seeds)) %>%
  `colnames<-`(c("data","criterion","geom_threshold","warm_start","seed")) %>%
  left_join(., df_match, by = c("data", "criterion", "geom_threshold", "warm_start", "seed")) %>%
  arrange(data, criterion, geom_threshold, warm_start, seed)

write.csv(job_status, "../scaling_job_status.csv", row.names = FALSE)

# write.csv(subset(job_status, is.na(runtime) & data != "EngyTime"), "failed_parameters.csv", row.names = FALSE)
# write.csv(subset(job_status, is.na(runtime) & data == "EngyTime"), "failed_parameters_engytime.csv", row.names = FALSE)


### Runtime
runtime_avgs <- df_match %>% filter(data != "EngyTime") %>%
  mutate(warm_start = if_else(warm_start == "oct", "K-means", "None")) %>%
  group_by(criterion, geom_threshold, warm_start) %>%
  summarize(result_cnt = n(),
            avg_runtime = mean(runtime/60)) 

df_match %>% filter(data != "EngyTime") %>%
  filter(criterion == "silhouette") %>% 
  group_by(geom_threshold, warm_start) %>%
  summarize(result_cnt = n(),
            avg_runtime = mean(runtime/60)) 

pal <- brewer.pal(n = 6, "Blues")

# sil_plot <- 
  runtime_avgs %>%
  filter(criterion == "dunnindex") %>%
  ggplot(aes(x = as.factor(geom_threshold), y = avg_runtime, color = warm_start, group = warm_start)) + 
  geom_line() + 
  scale_colour_manual(values = c(pal[4], pal[6])) +
  ggtitle("Effect of Scaling Methods on Algorithm Runtime", subtitle = "Dunn Index") + 
  labs(x = "Geometric Search Threshold (T)", y="Average Runtime (Minutes)", color = "Warm Start") + 
  theme(text=element_text(family="serif"))

dunn_plot <- runtime_avgs %>%
  filter(criterion == "dunnindex") %>%
  mutate("Warm Start" = warm_start) %>%
  ggplot(aes(x = as.factor(geom_threshold), y = avg_runtime, color = `Warm Start`, group = `Warm Start`)) + 
  geom_line() + 
  scale_colour_manual(values = c(pal[4], pal[6])) +
  ggtitle("Effect of Scaling Methods on Algorithm Runtime", subtitle = "Dunn Index") + 
  labs(x = "Geometric Search Threshold (T)", y="Average Runtime (Minutes)") + 
  theme(text=element_text(family="serif"))


################ Find score difference in fully scaled vs. unscaled
score_impact_sil <- df %>% filter(method == "ICOT_local" & criterion == "silhouette") %>%
  mutate(scaling_level = if_else((warm_start == "oct" & geom_threshold == .99), "FullScaled",
                                 if_else((warm_start == "none" & geom_threshold == 0), "Baseline", "Neither"))) %>%
  filter(scaling_level != "Neither" & data != "EngyTime") %>%
  group_by(data, criterion, scaling_level) %>%
  summarize(score_icot = mean(silhouette)) %>%
  spread(scaling_level, score_icot) %>%
  mutate("Score Change" = paste0(round((FullScaled - Baseline)/Baseline,3)*100,"%")) %>%
  rename("Baseline Score" = Baseline, "Fully Scaled Score" = FullScaled) 

score_impact_dunn <- df %>% filter(method == "ICOT_local" & criterion == "dunnindex") %>%
  mutate(scaling_level = if_else((warm_start == "oct" & geom_threshold == .99), "FullScaled",
                                 if_else((warm_start == "none" & geom_threshold == 0), "Baseline", "Neither"))) %>%
  filter(scaling_level != "Neither" & data != "EngyTime") %>%
  group_by(data, criterion, scaling_level) %>%
  summarize(score_icot = mean(dunn)) %>%
  spread(scaling_level, score_icot) %>%
  mutate("Score Change" = paste0(round((FullScaled - Baseline)/Baseline,3)*100,"%")) %>%
  rename("Baseline Score" = Baseline, "Fully Scaled Score" = FullScaled) 


score_impact %>% 
  group_by(metric) %>%
  summarize(avg_baseline = mean(`Baseline Score`), avg_scaled = mean(`Fully Scaled Score`),
            avg_loss = mean(`Score Change`))

score_impact
