setwd("~/Packages/Optimal_Clustering_Trees/Testing_Class_Project/data")
library(factoextra)

#Atom dataset
df <- read.table("Atom.lrn", quote="\"", comment.char="%")
df_class <- read.table("Atom.cls", quote="\"", comment.char="%")
data<-as.data.frame(cbind(df[,2:ncol(df)], label = df_class$V2))
write.csv(data,"atom.csv",row.names = FALSE)

#Hepta dataset
df <- read.table("Hepta.lrn", quote="\"", comment.char="%")
df_class <- read.table("Hepta.cls", quote="\"", comment.char="%")
data<-as.data.frame(cbind(df[,2:ncol(df)], label = df_class$V2))
write.csv(data,"Hepta.csv",row.names = FALSE)

#Target dataset
df <- read.table("Target.lrn", quote="\"", comment.char="%")
df_class <- read.table("Target.cls", quote="\"", comment.char="%")
data<-as.data.frame(cbind(df[,2:ncol(df)], label = df_class$V2))
write.csv(data,"Target.csv",row.names = FALSE)

#Tetra dataset
df <- read.table("Tetra.lrn", quote="\"", comment.char="%")
df_class <- read.table("Tetra.cls", quote="\"", comment.char="%")
data<-as.data.frame(cbind(df[,2:ncol(df)], label = df_class$V2))
write.csv(data,"Tetra.csv",row.names = FALSE)

#Two Diamonds dataset
df <- read.table("TwoDiamonds.lrn", quote="\"", comment.char="%")
df_class <- read.table("TwoDiamonds.cls", quote="\"", comment.char="%")
data<-as.data.frame(cbind(df[,2:ncol(df)], label = df_class$V2))
write.csv(data,"TwoDiamonds.csv",row.names = FALSE)

#Two WingNut
df <- read.table("WingNut.lrn", quote="\"", comment.char="%")
df_class <- read.table("WingNut.cls", quote="\"", comment.char="%")
data<-as.data.frame(cbind(df[,2:ncol(df)], label = df_class$V2))
write.csv(data,"WingNut.csv",row.names = FALSE)

#Lsurn
df <- read.table("Lsun.lrn", quote="\"", comment.char="%")
df_class <- read.table("Lsun.cls", quote="\"", comment.char="%")
data<-as.data.frame(cbind(df[,2:ncol(df)], label = df_class$V2))
write.csv(data,"Lsun.csv",row.names = FALSE)



