library(multcomp)
library(ggplot2)

##### Exercise 1 #####

num = c(10, 20, 30, 40)
contrMat(num, type = "Changepoint")

#           1       2       3      4
# C 1 -1.0000  0.2222  0.3333 0.4444
# C 2 -0.3333 -0.6667  0.4286 0.5714
# C 3 -0.1667 -0.3333 -0.5000 1.0000

#      mu1 != mu2 = mu3 = mu4 (-n1/n1, n2/(n2+n3+n4), n3/(n2+n3+n4), n4/(n2+n3+n4))
# H1 : mu1 = mu2 != mu3 = mu4 (-n1/(n1+n2), -n2/(n1+n2), n3/(n3+n4), n4/(n3+n4))
#      mu1 = mu2 = mu3 != mu4 (-n1/(n1+n2+n3), -n2/(n1+n2+n3), -n3/(n1+n2+n3), n4/n4

# for detecting where is the change point

##### Exercise 2 #####

# data

w1 = c(8.22, 8.34, 7.31, 6.13, 6.09, 7.01, 7.48, 6.96, 6.46, 6.27)
w2 = c(7.89, 6.68, 6.80, 6.81, 6.77, 6.98, 6.49, 8.03, 6.29, 7.33)
w3 = c(7.21, 8.96, 8.85, 7.95, 9.47, 8.32, 7.59, 8.22, 6.73, 7.95)
w4 = c(7.64, 7.19, 7.28, 8.47, 8.31, 9.41, 9.63, 9.14, 7.90, 7.78)

w = c(w1,w2,w3,w4)
n = c(length(w1), length(w2), length(w3), length(w4))
grp = factor(c(rep(1:4,n)))
weights = data.frame(w=w,grp=grp)

# 1. Specify the contrast matrix of interest along with its pattern of the alternative.

C = contrMat(n, type = "Changepoint")


# 2 

ggplot(weights, aes(x=grp, y=w, fill=grp)) + 
  geom_boxplot()+
  labs(x="Group", y = "Weight")+
  scale_fill_brewer(palette="Blues") + theme_classic()


# 3. Estimate the means and variances of the data per group.

wbar = aggregate(w~grp, data=weights, mean)[,2] #mean
wvar = aggregate(w~grp, data=weights, var)[,2] #variance


# 4. Test the null hypothesis formulated above with a multiple contrast test approach

Shat = diag(wvar/n)
Gammahat = C%*%Shat%*%t(C)

diff = C%*%wbar
Tl = diff/sqrt(c(diag(Gammahat)))

nul = sapply(1:nrow(C),function(arg){
  c(t(C[arg,])%*%Shat%*%C[arg,])^2/
    sum(C[arg,]^4*wvar^2/(n^2*(n-1)))})

T0 = max(abs(Tl))

### multivariate T

nu = round(min(nul))
Rhat = cov2cor(Gammahat)

set.seed(1)
tmax=qmvt(0.95, tail="both", corr=Rhat, df=nu)$quantile
T0>=tmax

pv<-sapply(1:3,function(j)
  1-pmvt(-abs(Tl[j]), abs(Tl[j]), df=nu, delta=rep(0,3), corr=Rhat)[1])

# pv = 0.0471551134 0.0008861882 0.0296193245


### bootstrap

Al = c()
nboot = 10000
s2vec = rep(wvar,n)

for(h in 1:nboot){
  XB <- rnorm(sum(n),0,sqrt(s2vec))
  DatB<-data.frame(XB=XB,grp=grp)
  XbarB<-aggregate(XB~grp,data=DatB,mean)[,2]
  si2B <-aggregate(XB~grp,data=DatB,var)[,2]
  ShatB <- diag(si2B/n)
  GammahatB<-C%*%ShatB%*%t(C)
  diffB <-C%*%XbarB
  TlB<-diffB/sqrt(c(diag(GammahatB)))
  Al[h]<-max(abs(TlB))
  }

pboot = mean(Al>=T0)

# pboot = 6e-04

# We can reject H0 for both methods.


# 5. Why is the parametric Bootstrap approach valid? 
# Estimate the correlation matrix of the resampling test statistics T* and compare with your computation of R.

# R

s2<- sum((n-1)*wvar)/(sum(n)-4)

Gamma= matrix(0,ncol=3,nrow=3)
for(i in 1:3){
  for(j in 1:3){
    if(i==j){Gamma[i,j]=s2*(1/n[1]+1/n[j+1])}
    if(i!=j){Gamma[i,j]=s2*(1/n[1])}
  }
}

R<-cov2cor(Gamma)

#      [,1] [,2] [,3]
# [1,]  1.0  0.5  0.5
# [2,]  0.5  1.0  0.5
# [3,]  0.5  0.5  1.0

# T*

RhatB = cov2cor(GammahatB)

#           C 1       C 2       C 3
# C 1 1.0000000 0.6975229 0.3210049
# C 2 0.6975229 1.0000000 0.4375255
# C 3 0.3210049 0.4375255 1.0000000


##### Exercise 3 #####

simu<-function(nsim, n, s1, s2, s3, s4, dist, a){
  
  num = c(n,n,n,n)
  N = sum(num)
  si2 = c(s1, s2, s3, s4)
  s2vec = rep(si2,num)
  C = contrMat(num, type = "Changepoint")
  
  if(dist == "Normal") {
    x <- matrix(rnorm(n = N*nsim, mean = 0, sd = s2vec), ncol = nsim)
  }
  if(dist == "Exp") {
    x <- matrix(((rexp(n = N*nsim, rate = 1) - 1) / sqrt(s2vec)), ncol = nsim)
  }
  if(dist=="T3"){
    x <- matrix(rt(n = N*nsim, N-a), ncol = nsim)
  }
  
  pvalue = c()
  grp = factor(c(rep(1:4,num)))
  
  for (i in 1:nsim){
    #cat("i : ", i, "\n")
    dat = data.frame(x=x[,i], grp=grp)
    xbar = aggregate(x~grp, data=dat, mean)[,2] #mean
    xvar = aggregate(x~grp, data=dat, var)[,2] #variance
    
    #cat("xbar and xvar : ", xbar, ", ", xvar, "\n")
    
    Shat = diag(xvar/n)
    Gammahat = C%*%Shat%*%t(C)
    
    diff = C%*%xbar
    Tl = diff/sqrt(c(diag(Gammahat)))
    
    #cat("Tl : ", Tl, "\n")
    
    nul = sapply(1:nrow(C),function(arg){
      c(t(C[arg,])%*%Shat%*%C[arg,])^2/
        sum(C[arg,]^4*wvar^2/(n^2*(n-1)))})
    
    nu = round(min(nul))
    Rhat = cov2cor(Gammahat)
    
    pv<-sapply(1:3,function(j)
      1-pmvt(-abs(Tl[j]), abs(Tl[j]), df=nu, delta=rep(0,3), corr=Rhat)[1])
    
    #cat("pv : ", pv, "\n")
    
    pvalue[i] = min(pv)
  }
  
  if(si2[1]==1){si2_name = 1}
  if(si2[1]!=1){si2_name = "from weights data"}
  
  result <- data.frame(n = n, s2 = si2_name, Dist = dist, pv = mean(pvalue))
  write.table(result, sep = "\t",  eol = "\r\n", row.names = FALSE, col.names = FALSE, quote = FALSE, append = TRUE)
  
}

dist = c("Normal", "Exp", "T3")
n = c(10, 20)
si = c(1,1,1,1)
s2 = array(c(si, wvar), dim = c(4, 2))

for (i in dist){
  for (j in n){
    for (k in 1:2){
      simu(10000, j, s2[1,k], s2[2,k], s2[3,k], s2[4,k], i, 4)
    }
  }
}
#  n                 s2   dist             pvalue
# 10	                1	Normal	0.494569243646206
# 10	from weights data	Normal	0.549289082796529
# 20	                1	Normal	0.497571757509778
# 20	from weights data	Normal	0.513968744001618
# 10	                1	   Exp	0.488058615976167
# 10	from weights data	   Exp	0.479412379380511
# 20	                1	   Exp	0.486037918573138
# 20	from weights data	   Exp	0.479903427124965
# 10	                1	    T3	0.490728079294779
# 10	from weights data	    T3	0.493661079097505
# 20	                1	    T3	0.500059757147593
# 20	from weights data	    T3	0.495309962772656

