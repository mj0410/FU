##### Exercise 1

myPermuCI<-function(nsim,nperm,n1,n2,v1,v2,delta, Distribution){
  PermCI=c()
  N<-n1+n2
  
  #------Data Generation-----#
  vvec = sqrt(c(rep(v1,n1),rep(v2,n2)))
  if(Distribution == "Normal"){
    x1=matrix(rnorm(n1*nsim, delta)*sqrt(v1), ncol=nsim)
    x2=matrix(rnorm(n2*nsim)*sqrt(v2), ncol=nsim)}
  if(Distribution == "Exp"){
    x1=matrix(rexp(n1 * nsim)-1+delta, ncol=nsim)
    x2=matrix(rexp(n2 * nsim)-1, ncol=nsim)
  }
  xy = rbind(x1,x2)
  x12 = x1^2; x22=x2^2
  mx = colMeans(x1); my = colMeans(x2)
  vx = (colSums(x12)-n1*mx^2)/(n1-1)
  vy = (colSums(x22)-n2*my^2)/(n2-1)
  df=(vx/n1+vy/n2)^2/(vx^4/(n1^2*(n1-1))+vy^4/(n2^2*(n2-1)))
  T.L <-mx-my-qt(0.975,df)*sqrt(vx/n1+vy/n2)
  T.U <-mx-my+qt(0.975,df)*sqrt(vx/n1+vy/n2)
  
  #-----------Permutation Matrices--------------#
  P<-t(apply(matrix(1:N,nrow=nperm,ncol=N,byrow=TRUE),1,sample))
  
  #-------Helping Variables for Permutation Distribution---#
  i1<-c(rep(1/n1,n1),rep(0,n2))
  i2<-c(rep(0,n1),rep(1/n2,n2))
  i3<-c(rep(1/(n1*(n1-1)),n1), rep(0,n2))
  i4<-c(rep(0,n1), rep(1/(n2*(n2-1)),n2))
  Im1<-matrix(i1[P],nrow=nperm,ncol=N)
  Im2<-matrix(i2[P],nrow=nperm,ncol=N)
  Iv1<-matrix(i3[P],nrow=nperm,ncol=N)
  Iv2<-matrix(i4[P],nrow=nperm,ncol=N)
  
  for (i in 1:nsim){
    X<-xy[,i]
    #permutation
    mxP <- Im1%*%X
    myP = Im2%*%X
    vxP <- Iv1%*%X^2 - n1/(n1*(n1-1))*mxP^2
    vyP <- Iv2%*%X^2 - n2/(n2*(n2-1))*myP^2
    TP = (mxP -myP )/sqrt(vxP +vyP )
    c1<-quantile(TP,0.025); c2<-quantile(TP,0.975)
    lower <-mx[i]-my[i]-c2*sqrt(vx[i]/n1+vy[i]/n2)
    upper <- mx[i]-my[i]-c1*sqrt(vx[i]/n1+vy[i]/n2)
    PermCI[i]<-(lower<delta& upper >delta)
  }
  
  result <- data.frame(nsim=nsim,nperm=nperm,delta=delta,
                       n1=n1,n2=n2,v1=v1,v2=v2, CI=mean(T.L <delta & T.U >delta),
                       PermCI=mean(PermCI),
                       distribution=Distribution)
  write.table(result, row.names = FALSE, col.names = FALSE, quote = FALSE, append = TRUE)
  return(result)
}

n=c(10,20,30)
delta=seq(0, 2, by = 0.1)
Dist=c("Normal", "Exp")

for(n1 in 1:length(n)){
  for(n2 in 1:length(n)){
    for(j in 1:length(delta)){
      for(k in 1:length(Dist)){
        myPermuCI(10000, 10000, n[n1], n[n2], 1, 1, delta[j], Dist[k])
      }
    }
  }
}



##### Exercise 2

# W = random weight / Z = centered sample / Xstar = resampled variables
# Tstar = resampled distribution / n=number of variables

##### one sample

# Assume that we have sample X, mX=mean of X, vX=variance of X

# Z=X-mX, Xstar=W*Z
# mXstar=mean of Xstar, vXstar=variance of Xstar
# Tstar=sqrt(n)*mXstar/sqrt(vXstar) => E(mxstar|X) = 0 since we centered variables

# c=quantile(Tstar, 1-alpha/2)
# CI = [mxstar-c*sqrt(vxstar)/sqrt(n), mxstar+c*sqrt(vxstar)/sqrt(n)]


##### two samples

# Assume that we have sample X1 with n1 variables, X2 with n2 variables
# mX1 = mean of X1, mX2 = mean of X2, vX1 = variance of X1, vX2 = variance of X2

# Z1 = X1-mX1, Z2 = X2-mX2, X1star = W*Z1, X2star = W*Z2
# mX1star = mean of X1star, mX2star = mean of X2sta
# vX1star = variance of X1star, vX2star = variance of X2star
# Tstar = (mx1star-mx2star)/sqrt(vX2star+vX2star)  => E(mx1star-mx2star|X) = 0 since we centered variables

# lower = quantile(Tstar, alpha/2), upper = quantile(Tstar, 1-alpha/2)
# CI = [mX1star-mX2star-upper*sqrt((vXstar1/n1)+(vX2star/n2)), mX1star-mX2star-lower*sqrt((vXstar1/n1)+(vX2star/n2))]




##### Exercise 3
##### SO CUTE THANK YOU :)

rm(list = ls())
library(ggplot2)
# create data
x <- c(8,7,6,7,6,5,6,5,4,5,4,3,4,3,2,3,2,1,0.5,0.1)
dat1 <- data.frame(x1 = 1:length(x), x2 = x)
dat2 <- data.frame(x1 = 1:length(x), x2 = -x)
dat1$xvar <- dat2$xvar <- NA
dat1$yvar <- dat2$yvar <- NA
dat1$siz <- dat2$siz <- NA
dat1$col <- dat2$col <- NA
dec_threshold = -0.5
set.seed(2512)
for (row in 1:nrow(dat1)){
  if (rnorm(1) > dec_threshold){
    dat1$xvar[row] <- row
    dat1$yvar[row] <- sample(1:dat1$x2[row]-1,1)
    dat1$siz[row] <- runif(1,0.5,1.5)
    dat1$col[row] <- sample(1:5, 1)
  }
  if (rnorm(1) > dec_threshold){
    dat2$xvar[row] <- row
    dat2$yvar[row] <- sample(1:dat2$x2[row],1)
    dat2$siz[row] <- runif(1,0.5,1.5)
    dat2$col[row] <- sample(1:5, 1)
  }
}
# plot the christmas tree
ggplot() +
  geom_bar(data = dat1, aes(x=x1, y=x2),stat = "identity", fill = '#31a354') +
  geom_bar(data = dat2, aes(x=x1, y=x2),stat = "identity", fill = '#31a354') +
  geom_point(data = dat1,aes(x = xvar, y = yvar, size = siz, colour = as.factor(col)) ) +
  geom_point(data = dat2,aes(x = xvar, y = yvar, size = siz, colour = as.factor(col)) ) +
  coord_flip() + theme_minimal()+ theme(legend.position="none",
                                        axis.title.x=element_blank(),
                                        axis.text.x=element_blank(),
                                        axis.ticks.x=element_blank(),
                                        axis.title.y=element_blank(),
                                        axis.text.y=element_blank(),
                                        axis.ticks.y=element_blank()) +
  ggtitle('We wish you a Merry Christmas')
