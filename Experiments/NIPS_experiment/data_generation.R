library(dplyr)
setwd("~/Packages/Optimal_Clustering_Trees/Experiments/NIPS_experiment")

dataset<-read.csv("general_data.csv")
 
df<-dataset %>%
  group_by(outcome)

patients<-unique(df$PID)
dat<-df[1:5,]
k=500
for (i in 1:k){
  #i=1
  x<-df[which(df$PID==patients[i]),] 
  dat<-rbind(as.data.frame(dat),as.data.frame(x[1,]))
}

x<-dat[6:(k+5),]
x$gender<-"male"
x$gender[which(x$Gender==2)]<-"female"
columns<-c("gender","diabetes","Age","SBP","HDL","BMI")
#columns<-c("gender","Glucose_bl","Age","SBP","HDL","BMI")
#columns<-c("Hemat","Age","SBP","HDL","BMI")
data_f<-x[,columns]
write.csv(data_f,"data.csv",row.names = FALSE)

#Effort number 2
dataset<-read.csv("framingham.csv")
data_b<-dataset[1:203,]
columns<-c("male","education","currentSmoker","BPMeds","prevalentHyp","prevalentStroke","diabetes")

data_c<-na.omit(data_b[,columns])
write.csv(data_c,"data.csv",row.names = FALSE)


