##### Exercise 1

before=c(8,7,7,4,6,3,6,6,3,9,4,7,5,3,8)
after=c(6,6,6,6,3,6,3,5,2,2,1,1,4,6,1)
data=c(before,after)

### 1 ###

library(ggplot2)

df=data.frame(p=c(seq(1,15,1),seq(1,15,1)),data=data,time=c(rep("before",15),rep("after",15)))

ggplot(df, aes(x=time, y=data, fill=time)) +
  geom_boxplot() +
  scale_x_discrete(limits=c("before","after"))

ggplot(df, aes(x=p, y=data, group=time)) +
  geom_line(aes(color=time))+
  geom_point(aes(color=time))

# According to the box plot, the treatment has an effect in general.
# The line plot, however, there are some patients who reported high scale after treatment.


### 2 ###

# There is difference between the data. The scales of most patients became lower after 3 months.


### 3 ###
 
# median

mb=median(before)
ma=median(after)

theta = mb-ma

# theta = 2
# It means there is difference between 'before' and 'after'

#proportion

grid=expand.grid(before,after)
pprop=(grid[,1]<grid[,2]) + 0.5*(grid[,1]==grid[,2])
p=mean(pprop)

# p=0.26 / It means 'before' tends to be larger than 'after' and thus there is an treatment effect.


### 4 ###

perm<-function(x,y){
  
  n<-length(x)
  
  #-------------Brunner-Munzel test-----#
  data<-c(x,y)
  rxy<-rank(data)
  mRx<-mean(rxy[1:n])
  mRy<-mean(rxy[(n+1):(2*n)])
  phat<-1/(2*n)*(mRy-mRx)+1/2
  
  rx <-rank(x)
  ry <-rank(y)
  Z1k <- 1/n*(rxy[1:n]-rx)
  Z2k <- 1/n*(rxy[(n+1):(2*n)]-ry)
  Dk <- Z1k-Z2k
  sigmahat <-var(Dk)
  
  T=sqrt(n)*(phat-1/2)/sqrt(sigmahat)
  pvalue=2*min(pt(T,n-1),1-pt(T,n-1))
  crit<-qt(0.975,n-1)
  
  #-------Building Permutation matrix---#
  nperm<-2^n
  p<-0
  
  for (i in 1:n){
      a<-rep(c(rep(c(i,i+n),nperm/(2^i)),rep(c(i+n,i),nperm/(2^i))),2^(i-1))
      p<-rbind(p,a)
    }
  p<-p[2:(n+1),]
  P<-matrix(p,ncol=nperm)
  
  
  #------Permutation----#
  BM=PERM=c()
  xperm<-matrix(data[P],nrow=(2*n),ncol=nperm)
  rxperm<-matrix(rxy[P],nrow=(2*n),ncol=nperm)
  
  xperm1<-xperm[1:n,]
  xperm2<-xperm[(n+1):(2*n),]
  
  rperm1<-rxperm[1:n,]
  rperm2<-rxperm[(n+1):(2*n),]
  
  riperm1<-apply(xperm1,2,rank)
  riperm2<-apply(xperm2,2,rank)
  
  BMperm2<-1/n*(rperm2-riperm2)
  BMperm3<-1/n*(rperm1-riperm1)-BMperm2
  
  pdperm<-colMeans(BMperm2)
  mperm3<-colMeans(BMperm3)
  vperm3<-(colSums(BMperm3^2)-n*mperm3^2)/(n-1)
  vperm30<-(vperm3==0)
  vperm3[vperm30]<-1/n
  
  Tperm<-sqrt(n)*(pdperm-1/2)/sqrt(vperm3)
  
  p1perm<-mean(Tperm<=T); p2perm<-mean(Tperm>=T)
  pperm<-2*min(p1perm,p2perm)

  result<-data.frame(nperm=nperm,n=n,BM=pvalue,PERM=pperm)
  result
  }

perm(before,after)

# nperm  n         BM       PERM
# 32768 15 0.02087329 0.03399658

# Both results from Munzel test and permuted version are smaller than 0.05, therefore we can reject H0.



##### Exercise 2

library(multcomp)

estimate<-function(n,m1,m2,s1,s2,cov){
  sigma=matrix(c(s1,cov,cov,s2),ncol=2)
  x=rmvnorm(n=n,mean=c(m1,m2),sigma=sigma)
  
  grid=expand.grid(x[,1],x[,2])
  prop=(grid[,1]<grid[,2]) + 0.5*(grid[,1]==grid[,2])
  p=mean(prop)
  
  if(p<1/2) print("p is smaller than 1/2")
  if(p==1/2) print("p is equal to 1/2")
  if(p>1/2) print("p is larger than 1/2")
}

estimate(100,0,0,1,1,0)
# "p is smaller than 1/2"
estimate(100,0,0,1,1,0.5)
# "p is smaller than 1/2"
estimate(100,0,0,1,2,0)
# "p is larger than 1/2"
estimate(100,1,0,1,1,0)
# "p is smaller than 1/2"



##### Exercise 3

# I couldn't find proper estimator so just calculated sign effect xi

data=cbind(before,after)
xprop=(data[,1]<data[,2]) + 0.5*(data[,1]==data[,2])
xi=mean(xprop)


