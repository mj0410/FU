##### Exercise 2 #####

library(multcomp)
library(dplyr)
data(recovery)
recovery

# 1. Can we conclude that any of the three blanket types b1; b2, or b3 leads to a significant reduction in recovery time 
# compared with the standard blanket b0 at level alpha = 5%? To do so, please follow these steps:

# (a) Define a family of null hypotheses.

# mu0,1,2,3 : mean of b0,1,2,3
# H0(0j) : mu0 = muj, j=1,2,3
# sigma = {H0(01), H0(02), H0(03)}


# (b) Calculate the corresponding test statistics.

bm <- aggregate(minutes~blanket, data=recovery, mean)[,2]
bsi2 <- aggregate(minutes~blanket, data=recovery, var)[,2]
num <- count(recovery, blanket)
n <- c(num[1,2], num[2,2], num[3,2], num[4,2])

N = sum(n); a=length(num[,2])

s2 <- sum((n-1)*bsi2)/(N-a)
s1j <- sapply(2:4, function(j)s2*(1/n[1] + 1/n[j]))

diff <-sapply(2:4,function(j)(bm[1]-bm[j]))
T1j <- diff/sqrt(s1j)
T0 <- max(abs(T1j))


# (c) Calculate Bonferroni-adjusted p-values.

pv <- sapply(2:4, function(j)2*min(pt(T1j[j-1],n[1]+n[j]-2), 1-pt(T1j[j-1],n[1]+n[j]-2)))
bonferroni_p <- pv*3

# 0.5931716252 0.0004069726 0.2053183050


# (d) Calculate the correlation matrix of the test statistics.

gamma = matrix(1, ncol=3, nrow=3)

for(i in 1:3){
  for(j in 1:3){
    if(i==j) {gamma[i,j]=s2*(1/n[1]+1/n[j+1])}
    if(i!=j) {gamma[i,j]=s2*1/n[1]}
  }
}

R = cov2cor(gamma)


# (e) Calculate the p-values using the Multiple Contrast Test Procedure.

pv_MCTP <- sapply(1:3, function(j)
                  1-pmvt(-abs(T1j[j]), abs(T1j[j]), df=N-a, delta=rep(0,3), corr=R)[1])

# 0.4559362595 0.0001169635 0.1820056433


# 2. Compute 95% two-sided simultaneous confidence intervals for each mean difference using the MCTP. 
# Display the confidence intervals graphically and interpret the results.

q = qmvt(0.95, tail="both", corr=R, df=N-4)$quantile
CI_MCTP <- sapply(1:3, function(j)q*s2*sqrt(1/n[1]+1/n[j+1]))

CI01 <- c(diff[1]-CI_MCTP[1], diff[1]+CI_MCTP[1])
# -8.201834, 12.468500

CI02 <- c(diff[2]-CI_MCTP[2], diff[2]+CI_MCTP[2])
# -2.86850, 17.80183

CI03 <- c(diff[3]-CI_MCTP[3], diff[3]+CI_MCTP[3])
# -4.035010, 7.368343

amod <- aov(minutes~blanket, data=recovery)
erg<-glht(amod,linfct=mcp(blanket="Dunnett"))
plot(erg)

# Confidence intervals of "b1-b0" and "b3-b0" contain 0 and "b2-b0" doesn't contain 0.
# Therefore we can conclude that the blanket b2 has significant effect for recovery compared to other blankets.
