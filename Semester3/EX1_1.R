################################
########## EXERCISE 2 ##########
################################

# I tried to find out how to apply alternative hypothesis mean=delta, where delta is part of {0, 0.1, 0.2, ... , 1}
# however I failed to figure it out, therefore I set null hypothesis as H0:mu=0 and simulate the estimating type 1 error rate.

library(pwr)

n=c(5,10,20,30,50,100)
delta=seq(0, 1, by=0.1)

ex2=function(m, s2, nsim, dist){
  t1e=c()
  for(i in 1:length(m)){
    n=m[i]
    crit=qt(0.975, n-1)
    set.seed(1)
  
    if(dist=="normal"){
      x=matrix(rnorm(n*nsim)*sqrt(s2), nrow=n, ncol=nsim)
    }
    if(dist=="exp"){
      x=matrix((rexp(n*nsim)-1)*sqrt(s2), nrow=n, ncol=nsim)
    }
    if(dist=="log"){
      x=matrix(rlnorm(n*nsim)*sqrt(s2), nrow=n, ncol=nsim)
    }
    if(dist=="chi"){
      x=matrix((rchisq(n*nsim, n-1)-(n-1))*sqrt(s2), nrow=n, ncol=nsim)
    }
    if(dist=="uni"){
      x=matrix(runif(n*nsim, min=-1, max=1)*sqrt(s2), nrow=n, ncol=nsim)
    }
  
    mx=colMeans(x)
    sdx=sqrt((colSums(x^2)-n*mx^2)/(n-1)) #standard deviation
    T=sqrt(n)*mx/sdx
    ttest=(abs(T)>=crit) #reject H0(TRUE/FALSE)
    t1e[i]=mean(ttest)
    }
  result=data.frame(Distribution=dist, n=m, Type1Error=t1e)
  result
}

ex2(n,1,10000,"normal")
ex2(n,1,10000,"exp")
ex2(n,1,10000,"log")
ex2(n,1,10000,"chi")
ex2(n,1,10000,"uni")


# estimating power of the one sample t-test

# This solution is wrong one I guess, but I at least want to try to calculate power of the t-test
# and see impact of n on power.

power=function(m, d){
  pm=c(); p=c()
  for(i in 1:length(m)){
    n=m[i]
    for(j in 1:length(d)){
      #calculate power for each delta(0, 0.1, ..., 1)
      pm[j]=pwr.t.test(n=n, d=(0-d[j]), power=NULL, sig.level=0.05, type="one.sample")$power
    }
    #save mean of power of each sample size
    p[i]=mean(pm)
  }
  result=data.frame(n=m, power=p)
  result
}

power(n, d)



##################################
########## EXERCISE 3.1 ##########
##################################


simulation=function(n, s2, alpha){
  set.seed(1)
  x=rnorm(n)*sqrt(s2)
  mx=mean(x)
  sdx=sqrt((sum(x^2)-n*mx^2)/(n-1))
  crit=qt(1-(alpha/2),n-1)
  
  #confidence interval=[lower:upper]
  lower=mx-crit/sqrt(n)*sdx
  upper=mx+crit/sqrt(n)*sdx
  
  hist(x, freq=FALSE)
  lines(density(x))
  
  #calculate area under the density curve between upper and lower
  #area=ecdf(x)
  #area(upper)-area(lower)
  pnorm(upper,mean=mx,sd=s2)-pnorm(lower,mean=mx,sd=s2)
}

simulation(100, 1, 0.05)





##################################
########## EXERCISE 3.2 ##########
##################################


theta2=function(x, n, nsim){
  s=c()
  for(i in 1:nsim){
    a=0
    for(j in 1:n){
      for(k in 1:n){
        if(j != k){
          a = a+(x[,i][j]*x[,i][k])
        }
      }
    }
    s[i]=a
  }
  s
}

simu=function(n,s2,nsim,dist){
  if(dist=="normal"){
    x=matrix(rnorm(n*nsim)*sqrt(s2), ncol=nsim)
  }
  if(dist=="exp"){
    x=matrix((rexp(n*nsim)-1)*sqrt(s2), ncol=nsim)
  }
  
  mx=colMeans(x)
  v1=mx^2
  v2=theta2(x, n, nsim)/(n*(n-1))
  
  #expected value=sum(value*probability)
  mu=colSums(x*(1/n))
  ex=mu^2
  
  result=data.frame(n=n, sigma2=s2, Distribution=dist, bias.v1=mean(v1-ex), MSE.v1=mean((v1-ex)^2), bias.v2=mean(v2-ex), MSE.v2=mean((v2-ex)^2))
  result
}

simu(10, 1, 10, "normal")

