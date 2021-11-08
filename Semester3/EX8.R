##### Exercise 1 #####

x=c(79.1, 82.0, 80.2, 82.8, 82.0, 79.6, 80.2, 77.1, 79.3, 79.9, 81.8, 79.5, 80.3, 78.3, 81.8)
y=c(73.3, 74.7, 75.1, 77.4, 76.5, 75.3, 74.3, 69.1, 75.4, 75.9, 76.6, 74.0, 75.1, 72.7, 74.0)

# 1. Are data independent? Plot the bodyweights data (scatterplot) and interpret the result.

plot(x,y,pch=19,cex=1.3,col="#E7B800")

# 2.

mx=mean(x); my=mean(y)
zx=x-mx; zy=y-my

plot(zx,zy,pch=19,cex=1.3,col="#00AFBB")
abline(v=0,col="grey"); abline(h=0,col="grey")

# 3. Estimate the covariance matrix.

xy=cbind(x,y)
var(xy)

# 4. Test the null hypothesis 'no change of the bodyweights' using an appropriate method.

diff=x-y; n=length(diff)
md=mean(diff)
vd=var(diff)
T=sqrt(n)*md/sqrt(vd)
pvalue=2*min(pt(T,n-1), 1-pt(T,n-1))

t.test(x,y,paired=TRUE)

##### Exercise 2 #####

library(multcomp)

sim=function(n,nsim,nperm,s1,s2,rho){
  sigma=matrix(c(s1,rho,rho,s2),ncol=2)
  x=matrix(rmvnorm(n=n*nsim,mean=c(0,0),sigma=sigma), ncol=nsim)
  d=x[1:n,]-x[(n+1):(2*n),]
  md=colMeans(d); vd=apply(d, 2, var);
  T=sqrt(n)*md/sqrt(vd)
  
  P <- t(apply(matrix(1:(2*n), ncol = nperm, nrow = 2*n, byrow = TRUE), 1, sample))
  
  Tstar=c()
  for(i in 1:nsim){
    xi <- x[,i]
    xstar <- matrix(xi[P], ncol = nperm)
    dstar=xstar[1:n,]-xstar[(n+1):(2*n),]
    mdstar=colMeans(dstar); vdstar=apply(dstar, 2, var)
    Tperm=sqrt(n)*mdstar/sqrt(vdstar)
    
    Tstar[i] <- (T[i] > quantile(Tperm,0.975) | T[i] < quantile(Tperm,0.025))
  }
  
  result<-data.frame(n=n, nsim=nsim, nperm=nperm, cov=rho, perm=mean(Tstar))
  print(result)
}

# sim(10,10000,10000,1,1,0.5)

n=c(10,20)
rho=c(-0.95,-0.5,0,0.5,0.95)

for(i in n){
  for(j in rho){
    sim(i,10000,10000,1,1,j)
  }
}

## result
#
# n  nsim nperm   cov   perm
# 10 10000 10000 -0.95 0.0481
# 10 10000 10000 -0.5 0.0511
# 10 10000 10000   0 0.053
# 10 10000 10000 0.5 0.0502
# 10 10000 10000 0.95 0.0549
# 20 10000 10000 -0.95 0.0539
# 20 10000 10000 -0.5 0.0522
# 20 10000 10000   0 0.0471
# 20 10000 10000 0.5 0.05
# 20 10000 10000 0.95 0.0508