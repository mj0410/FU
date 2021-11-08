

sim <- function(Dist, weight, n, nsim, nboot) {
  
  T=Tboot=c()
  crit <- qt(0.975, n-1) # Critical value at 5% level
  
  # generate data
  if(Dist == "Normal") {
    x <- matrix(rnorm(n = n*nsim, mean = 0, sd = 1), ncol = nsim)
  }
  if(Dist == "Exponential") {
    x <- matrix(((rexp(n = n*nsim, rate = 1) - 1) / sqrt(1)), ncol = nsim)
  }
  if(Dist == "Lognormal") {
    x <- matrix(((exp(rnorm(n = n*nsim, mean = 0, sd = 1)) - exp(1/2)) / sqrt(exp(1)*(exp(1)-1))), ncol = nsim)
  }
  if(Dist == "Chisq10") {
    x <- matrix(((rchisq(n = n*nsim, df = 10) - 10) / sqrt(20)), ncol = nsim)
  }
  if(Dist == "Uniform") {
    x <- matrix(((runif(n = n*nsim, min = 0, max = 1) - 1/2) / sqrt(1/12)), ncol = nsim)
  }
  
  ### test statistic for sample ###
  mx <- colMeans(x)
  vx <- (colSums(x^2) - n*mx^2) / (n-1)
  T <- sqrt(n) * mx / sqrt(vx)
  
  ### wild bootstrap resampling ###
  for(i in 1:nsim){
    xstar <- matrix(nrow=n, ncol=nboot)
    z <- x[,i]-mx[i]
    for(j in 1:nboot){
      if(weight == "normal") w <- rnorm(n, 0, 1)
      if(weight == "rademacher") w <- rbinom(n,1,1/2)*2-1
      if(weight == "uniform") w <- runif(n, -sqrt(12)/2, sqrt(12)/2)
      
      xstar[,j] <- z*w
    }
    
    mxstar = colMeans(xstar)
    vxstar = (colSums(xstar^2) - n*mxstar^2)/(n-1)
    Tstar = sqrt(n)*mxstar/sqrt(vxstar)
    p1 = mean(Tstar >= T[i])
    p2 = mean(Tstar <= T[i])
    Tboot[i] = (2*min(p1,p2)<0.05)
  }
  
  result <- data.frame(n = n, Dist = Dist, Weight = weight, Tori = mean(abs(T) > crit), Twild = mean(Tboot))
  write.table(result, sep = "\t",  eol = "\r\n", row.names = FALSE, col.names = FALSE, quote = FALSE, append = TRUE)
  return(result)
  
}

Dist <- c("Normal", "Exponential", "Lognormal", "Chisq10", "Uniform")
weight <- c("normal", "rademacher", "uniform")
n <- c(5, 10, 20, 30, 50, 100)

for (i in 1:length(Dist)) {
  for (j in 1:length(weight)) {
    for (k in 1:length(n)) {
      sim(Dist[i], weight[j], n[k], 10000, 100)
    }
  }
}
