library(dplyr)
setwd("~/Packages/Optimal_Clustering_Trees/Experiments/NIPS_experiment")

dataset<-read.csv("general_data.csv")
 
df<-dataset %>%
  group_by(outcome)

patients<-unique(df$PID)
dat<-df[1:5,]
k=200
for (i in 1:k){
  #i=1
  x<-df[which(df$PID==patients[i]),] 
  dat<-rbind(as.data.frame(dat),as.data.frame(x[1,]))
}

dat<-dat[6:k+5,]
dat$gender<-"male"
dat$gender[which(dat$Gender==2)]<-"female"
columns<-c("gender","diabetes","Age","SBP","HDL","BMI")

data_f<-dat[,columns]
write.csv(data_f,"data.csv")

