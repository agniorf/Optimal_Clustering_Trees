library(data.table)
library(tidyverse)
library("RColorBrewer")
cbPalette <- c("#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7")


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
data <- c("Atom", "Chainlink",
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
  mutate(warm_start = if_else(warm_start == "oct", "K-means", "None"),
         geom_threshold = as.factor(geom_threshold)) %>%
  group_by(criterion, geom_threshold, warm_start) %>%
  summarize(result_cnt = n(),
            avg_runtime = mean(runtime/60),
            sd_runtime = sd(runtime/60)) 

df_match %>% filter(data != "EngyTime") %>%
  filter(criterion == "silhouette") %>% 
  group_by(geom_threshold, warm_start) %>%
  summarize(result_cnt = n(),
            avg_runtime = mean(runtime/60),
            sd_runtime = sd(runtime/60))

sil_plot <- runtime_avgs %>%
  filter(criterion == "silhouette") %>%
  ggplot(aes(x = geom_threshold, y = avg_runtime, color = warm_start, group = warm_start)) + 
  scale_colour_manual(values=cbPalette)+
  geom_line() +
  geom_point()+
  theme_light()+
  theme(text = element_text(size=15)) + 
  labs(x = "Geometric Search Threshold (T)", y="Average Runtime (Minutes)", color = "Warm Start")

dunn_plot <- runtime_avgs %>%
  filter(criterion == "dunnindex") %>%
  ggplot(aes(x = geom_threshold, y = avg_runtime, color = warm_start, group = warm_start)) + 
  scale_colour_manual(values=cbPalette)+
  geom_line() +
  geom_point()+
  theme_light()+
  theme(text = element_text(size=15)) + 
  labs(x = "Geometric Search Threshold (T)", y="Average Runtime (Minutes)", color = "Warm Start")

sil_plot
dunn_plot

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
