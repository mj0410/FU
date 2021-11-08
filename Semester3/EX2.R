ns=c(10, 15, 20)
oriT=c()

###store original T of each n (10,15,20)

for(n in ns){

  ###draw a random sample
  set.seed(1)
  
  X=rnorm(n, mean=0, sd=1)
  mx <- mean(X)
  sdx <- sd(X)

  ###calculate the test statistic
  
  oriT <- append(oriT, sqrt(n)*mx/sdx)
  
}



###t-test
###type1 error simulation study

type1error <- function(n, nsim, dist) {
  
  set.seed(1)
  crit <- qt(0.975, n-1)
  
  if(dist == "Normal") {
    x <- matrix(rnorm(n = n*nsim, mean = 0, sd = 1), ncol = nsim)
  }
  if(dist == "Exponential") {
    x <- matrix(((rexp(n = n*nsim, rate = 1) - 1) / sqrt(1)), ncol = nsim)
  }
  
  #calculate test statistic
  mx <- colMeans(x)
  vx <- (colSums(x^2) - n*mx^2) / (n-1)
  tTest <- sqrt(n) * mx / sqrt(vx)
  result <- data.frame(n = n, Dist = dist, tTest = mean(abs(tTest) > crit))
  write.table(result, row.names = FALSE, col.names = FALSE, quote = FALSE, append = TRUE)
  return(result)
  
}

dist=c("Normal", "Exponential")

ah=c()

#alpha hat for each simulation will be stored in ah

for(k in dist){
  for(i in ns){
    err = type1error(i, 10000, k)
    ah = append(ah, err$tTest)
  }
}



###Resampling

X=rnorm(10, mean=0, sd=1)

resampling=function(n, nsim, x){
  
  if(n==10)tn=1
  if(n==15)tn=2
  if(n==20)tn=3
  
  xR=matrix(sample(x, n*nsim, TRUE), ncol=nsim)
  mxR <- colMeans(xR)
  sdxR <-sqrt((colSums(xR^2)-n*mxR^2)/(n-1))
  TR <- sqrt(n)*mxR/sdxR
  
  c1 <- quantile(TR,0.025) #2.5%quantile
  c2 <- quantile(TR,0.975) #97.5%quantile
  
  p <- 2*min(mean(TR<=oriT[tn]), mean(TR>=oriT[tn]))
  
  crit <- qt(0.975, n-1)
  tTest <- mean(abs(TR) > crit)
  
  result=c(p, tTest)
  return(result)
}

p=c()
ah_re=c()
for(i in ns){
    re = resampling(i, 10000, X)
    p = append(p, re[1])
    ah_re = append(ah_re, re[2])
}


###p-value from t(n-1)

pvt=c()
for(i in 1:3){
  pvt = append(pvt, 2*min(pt(oriT[i], ns[i]-1), 1-pt(oriT[i], ns[i]-1)))
}


###precision interval

PI=function(alpha, nsim){
  pi=c(alpha-(1.96*sqrt(alpha*(1-alpha))/nsim), alpha+(1.96*sqrt(alpha*(1-alpha))/nsim))
  return(pi)
}

###evaluate estimated alpha

for(i in 1:6){
  if(i<4){
    cat("Normal distribution with n=", ns[i], "simulation : ")
  }
  if(i>=4){
    cat("Exponential distribution with n=", ns[i-3], "simulation : ")
  }
  
  pi=PI(0.05, 10000)
  
  if(ah[i]>=pi[1] & ah[i]<=pi[2]){
    cat("accurate\n")
  }
  else if(ah[i]>pi[2]){
    cat("liberal\n")
  }
  else{
    cat("conservative\n")
  }
}

