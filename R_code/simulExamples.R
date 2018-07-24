install.packages("pracma")
library(pracma)

createBall <- function(minx,miny,maxx,maxy,N){
  df<-as.data.frame(randp(N))
  df$V1<-(df$V1+1)/2
  df$V2<-(df$V2+1)/2
  df$V1<- (maxx-minx)*(df$V1-1) + maxx
  df$V2<- (maxy-miny)*(df$V2-1) + maxy
  return(df)
}

df1<-createBall(0.2,0,0.4,0.3,60)
df1$cluster <-1
df2<-createBall(0.45,0.1,0.65,0.35,60)
df2$cluster <-2
df3<-createBall(0.3,0.6,0.5,0.8,60)
df3$cluster <-3

df<-read.csv("/Users/agni/Packages/Optimal_Clustering_Trees/data/localSearch_7.csv")

df<-as.data.frame(rbind(df1,df2,df3))
plot(x = df$V1, y=df$V2)

write.csv(df, "/Users/agni/Packages/Optimal_Clustering_Trees/data/localSearch_7.csv", row.names = FALSE)
