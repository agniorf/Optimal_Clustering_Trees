setwd("~/Dropbox (Personal)/Clustering/data/FCPS/01FCPSdata")
#install.packages("factoextra")
library(factoextra)

#Lsun
Lsun <- read.table("~/Dropbox (Personal)/Clustering/data/FCPS/01FCPSdata/Lsun.lrn", quote="\"", comment.char="%")
Lsun_class <- read.table("~/Dropbox (Personal)/Clustering/data/FCPS/01FCPSdata/Lsun.cls", quote="\"", comment.char="%")

Lsun_big<-as.data.frame(cbind(Lsun, label = Lsun_class$V2))

df<- Lsun[,2:3]
k=3

clustering<-function(df,k){
  df <- scale(df)
  #fviz_nbclust(df, kmeans, method = "wss") +
  #  geom_vline(xintercept = 4, linetype = 2)
  set.seed(123)
  km.res <- kmeans(df, k, nstart = 25)
``
  df_r <- cbind(df, k_means_cluster = km.res$cluster)
  clusters <- hclust(dist(df))
  clusterCut <- cutree(clusters, k)
  df_r <- cbind(df_r, hierarch_cluster = clusterCut)
  return(df_r)
}
lsun<- clustering(df,k)
lsun_r<-as.data.frame(cbind(lsun, true_label = Lsun_class$V2, obs_id = Lsun$V1))

table(lsun_r$true_label,lsun_r$k_means_cluster)/nrow(lsun_r)
#Flip values because they are not aligned
lsun_r$k_means_cluster[which(lsun_r$k_means_cluster==1)]<-4
lsun_r$k_means_cluster[which(lsun_r$k_means_cluster==3)]<-1
lsun_r$k_means_cluster[which(lsun_r$k_means_cluster==4)]<-3


table(lsun_r$true_label,lsun_r$hierarch_cluster)/nrow(lsun_r)
#Flip values because they are not aligned
lsun_r$hierarch_cluster[which(lsun_r$hierarch_cluster==1)]<-4
lsun_r$hierarch_cluster[which(lsun_r$hierarch_cluster==2)]<-5
lsun_r$hierarch_cluster[which(lsun_r$hierarch_cluster==3)]<-2
lsun_r$hierarch_cluster[which(lsun_r$hierarch_cluster==5)]<-1
lsun_r$hierarch_cluster[which(lsun_r$hierarch_cluster==4)]<-3

#Now all the classes are aligned and we can calculate the accuracy of the clustering
sum(diag(table(lsun_r$true_label,lsun_r$k_means_cluster)))/nrow(lsun_r)
sum(diag(table(lsun_r$true_label,lsun_r$hierarch_cluster)))/nrow(lsun_r)

lsun_r[,1:2]<-Lsun[,2:3]
setwd("~/Dropbox (Personal)/Clustering/data/FCPS/05Results")

write.csv(lsun_r,"Lsun_hierarchical_k_means_k_3.csv",row.names = FALSE)





#Atom Dataset
Atom <- read.table("~/Dropbox (Personal)/Clustering/data/FCPS/01FCPSdata/Atom.lrn", quote="\"", comment.char="%")
Atom_class <- read.table("~/Dropbox (Personal)/Clustering/data/FCPS/01FCPSdata/Atom.cls", quote="\"", comment.char="%")

Atom_big<-as.data.frame(cbind(Atom, label = Atom_class$V2))

df<- Atom[,2:4]
k=2
atom<- clustering(df,k)
atom_r<-as.data.frame(cbind(atom, true_label = Atom_class$V2, obs_id = Atom$V1))

table(atom_r$true_label,atom_r$k_means_cluster)/nrow(atom_r)
#Flip values because they are not aligned
atom_r$k_means_cluster[which(atom_r$k_means_cluster==1)]<-3
atom_r$k_means_cluster[which(atom_r$k_means_cluster==2)]<-1
atom_r$k_means_cluster[which(atom_r$k_means_cluster==3)]<-2

table(atom_r$true_label,atom_r$hierarch_cluster)/nrow(atom_r)
#Flip values because they are not aligned
atom_r$hierarch_cluster[which(atom_r$hierarch_cluster==1)]<-3
atom_r$hierarch_cluster[which(atom_r$hierarch_cluster==2)]<-1
atom_r$hierarch_cluster[which(atom_r$hierarch_cluster==3)]<-2

#Now all the classes are aligned and we can calculate the accuracy of the clustering
sum(diag(table(atom_r$true_label,atom_r$k_means_cluster)))/nrow(atom_r)
sum(diag(table(atom_r$true_label,atom_r$hierarch_cluster)))/nrow(atom_r)

atom_r[,1:3]<-Atom[,2:4]
setwd("~/Dropbox (Personal)/Clustering/data/FCPS/05Results")

write.csv(atom_r,"Atom_hierarchical_k_means_k_2.csv",row.names = FALSE)


