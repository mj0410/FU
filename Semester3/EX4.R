####################################
###### ----- Exercise 1 ----- ######
####################################

sim <- function(Dist, n, nsim, nboot) {

  B<-apply(matrix(1:n, ncol=nboot, nrow=n), 2, sample, replace=TRUE)
  tauhat2 <- c(); tauemp2 <- c()
  
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
  
  # original variance
  theta <- var(colMeans(x))
  
  #bootstrap
  for(i in 1:nsim){
    xstar <- matrix(x[,i][B], ncol=nboot, nrow=n) # x[,i] at position B
    mxstar <- colMeans(xstar)
    vxstar = (colSums(xstar^2) - n*mxstar^2)/(n-1)
    
    #calculate estimator
    tauhat2[i] <- (sum(vxstar^2)-nboot*mean(vxstar))/(nboot-1)
    tauemp2[i] <- mean(vxstar/n)
  }
  
  result <- data.frame(n = n, Dist = Dist, bias.tauhat2=mean(tauhat2-theta), MSE.tauhat2=mean((tauhat2-theta)^2), bias.tauemp2=mean(tauemp2-theta), MSE.tauemp2=mean((tauemp2-theta)^2))
  write.table(result, sep = "\t",  eol = "\r\n", row.names = FALSE, col.names = FALSE, quote = FALSE, append = TRUE)
  return(result)
  
}

Dist <- c("Normal", "Exponential", "Lognormal", "Chisq10", "Uniform")
n <- c(5, 10, 20, 30, 50, 100)

for (i in 1:length(Dist)) {
  for (j in 1:length(n)) {
      sim(Dist[i], n[j], 10, 10)
  }
}


####################################
###### ----- Exercise 2 ----- ######
####################################

# We can calculate true variance using given data if it has number of simulation groups with two variables.
# The correlation coefficients will be able to be calculated from each simulation and then the variance of
# correlation coefficients also can be calculated.
# The bootstraping is a good way to estimate the variance and observe the performance of estimator.

