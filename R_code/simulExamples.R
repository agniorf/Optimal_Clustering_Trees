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

df1<-createBall(0,0,0.2,0.3,1000)
df2<-createBall(0.8,0.1,1,0.35,1000)
df3<-createBall(0.35,0.8,0.65,1,100)

df<-as.data.frame(rbind(df1,df2,df3))
plot(x = df$V1, y=df$V2)
