########## EXERCISE 3.1 ##########

simulation=function(n, s2, alpha){
  set.seed(1)
  x=rnorm(n)*sqrt(s2)
  mx=mean(x)
  sdx=sqrt((sum(x^2)-n*mx^2)/(n-1))
  crit=qt(1-(alpha/2),n-1)

  #confidence interval=[lower:upper]
  lower=mx-crit/sqrt(n)*sdx
  upper=mx+crit/sqrt(n)*sdx

  hist(x, freq=FALSE, main = "Normal DIstribution")
  lines(density(x))

  #calculate area under the density curve between upper and lower
  #area=ecdf(x)
  #area(upper)-area(lower)
  pnorm(upper,mean=mx,sd=s2)-pnorm(lower,mean=mx,sd=s2)
}

simulation(100, 1, 0.05)



########## EXERCISE 3.2 ##########

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
  
  result=data.frame(n=n, sigma2=s2, Distribution=dist, bias.v1=mean(v1-s2), MSE.v1=mean((v1-s2)^2), bias.v2=mean(v2-s2), MSE.v2=mean((v2-s2)^2))
  result
}

simu(10, 1, 10000, "normal")
