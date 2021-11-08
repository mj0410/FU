library(datarium)
library(ggplot2)
library(cowplot)

score=c(selfesteem$t1, selfesteem$t2, selfesteem$t3)
x=matrix(score, ncol=3)
data<-data.frame(ID=rep(1:10, 3), score=score, test=rep(c("t1", "t2", "t3"), each=10))



##########################################################################
##### 1. Display the data graphically using boxplots and line plots. #####
##########################################################################


box<-ggplot(data, aes(x=test, y=score, fill=test)) +
  geom_boxplot() +
  scale_fill_brewer(palette="Dark2") + theme_classic()

line<-ggplot(data, aes(x=ID, y=score, group=test)) +
  geom_line(aes(color=test))+
  geom_point(aes(color=test))+
  scale_color_brewer(palette="Dark2")

plot_grid(box, line, labels = "AUTO")




########################################################################
##### 2. Estimate the means and the covariance matrix of the data. #####
########################################################################


Xbar=colMeans(x)
Vhat=var(x)



#################################################################################################################
##### 3. Formulate the global null hypothesis of no treatment effect and ########################################
##### test the hypothesis with the Wald-type and the ANOVA-type test statistic at 5% level of significance. #####
##### Does the test decision answer the main question of the practitioners? #####################################
#################################################################################################################


# H0 : mu1=mu2=mu3

# wald-type

library(multcomp)
library(MASS)

n=10; d=5

C = contrMat(n=rep(10,3),"GrandMean")
CX = C%*%Xbar
CVhat = C%*%Vhat%*%t(C)
W = n*t(CX)%*%ginv(CVhat)%*%CX
wald_pvalue = 1-pchisq(W, d-1)

# ANOVA-type

TT<- t(C)%*%ginv(C%*%t(C))%*%C
TrTV <-sum(c(diag(TT%*%Vhat)))
A <- n*t(Xbar)%*%TT%*%Xbar/TrTV
TVTV<-TT%*%Vhat%*%TT%*%Vhat
TrTVTV <-sum(c(diag(TVTV)))
f <- TrTV^2/TrTVTV
ANOVA_pvalue <- 1-pf(A,f,10^10)

cat("wald-type : ", wald_pvalue, ", ANOVA-type : ", ANOVA_pvalue)

# wald-type :  0 , ANOVA-type :  0

#-----------------------------#

library(MANOVA.RM)

fit<-RM(score~test, subject="ID", data=data)
summary(fit)

# p-values of both test are smaller than 0.05. Therefore, we can reject H0.




#########################################################################################################################
##### 4. The researchers aim to compare all time points with each other. Formulate the contrast matrix of interest. #####
#########################################################################################################################


C = contrMat(n=rep(10,3),"Tukey")





#####################################################################################################
##### 5. Compute a multiple contrast test using a multivariate T(n-1; 0; Rhat) distribution and #####
##### compute 95% two-sided simultaneous confidence intervals for each mean difference. #############
##### Display the confidence intervals graphically and interpret the results. #######################
#####################################################################################################


si2 <- aggregate(score ~ test, var, data = data)[,2]

n <- rep(10,3)
Shat <- diag(si2/n)
Gammahat<-C%*%Shat%*%t(C)


# Differences
diff <- C %*% Xbar

# Test statistics
Tl<-diff/sqrt(c(diag(Gammahat)))

# Maximum of Test Statistics
T0<-max(abs(Tl))

# df = n-1
nu = 9

# Correlation Matrix
Rhat<-cov2cor(Gammahat)
set.seed(1)

# Quantile of maximum
tmax=qmvt(0.95,tail="both",corr=Rhat, df=nu)$quantile
T0>=tmax

# P value calculation
pv<-sapply(1:3,function(j) 1-pmvt(-abs(Tl[j]),abs(Tl[j]),df=nu, delta=rep(0,3),corr=Rhat)[1])

# Compute 95% CIs for each mean difference

s2 <- sum((n-1)*si2)/(30-3)
CI_l <- sapply(1:3, function(j) diff[j] - tmax * sqrt(s2) * sqrt(1/5))
CI_u <- sapply(1:3, function(j) diff[j] + tmax * sqrt(s2)* sqrt(1/5))

# Visualize the confidence intervals
data_CI <- data.frame(diff = diff, lower = CI_l, upper = CI_u)
ggplot(data = data_CI, aes(x = rownames(data_CI), y = diff, color = rownames(data_CI)))+
  geom_point()+
  geom_errorbar(aes(ymax = upper, ymin = lower))+
  xlab("Contrasts")+
  ylab("Difference")+
  ggtitle("Confidence intervals for mean differences")+
  theme(plot.title = element_text(hjust = 0.5))

# We can conclude there are significant differences between all time points.



################################################################################################################# 
#### 6. In addition, compute a multiple contrast test using nonparametric and parametric bootstrap methods. #####
#################################################################################################################

param_Al<-c(); non_Al<-c()
nboot <- 10000
s2vec <-rep(si2,n)
N <- sum(n)
group <- rep(c("t1", "t2", "t3"), each = 10)
param_TlB <- matrix(rep(0,3*nboot), ncol = nboot); non_TlB <- matrix(rep(0,3*nboot), ncol = nboot)

# Parametric bootstrap 
for(h in 1:nboot){
  XB <- sample(score, 30, replace=TRUE)
  DatB <- data.frame(XB=XB, grp=group)
  
  # Calculate means and variances
  XbarB <- aggregate(XB~group,data=DatB,mean)[,2]
  si2B <-aggregate(XB~group,data=DatB,var)[,2]
  ShatB <- diag(si2B/n)
  GammahatB <- C%*%ShatB%*%t(C)
  diffB <- C%*%XbarB
  
  # Test statistic
  non_TlB[,h] <- diffB/sqrt(c(diag(GammahatB)))
  non_Al[h] <- max(abs(non_TlB[,h]))
}

# pvalue
non_pvalue = mean(non_Al>=T0)


# Parametric bootstrap 
for(h in 1:nboot){
  # Draw data from Normal distribution
  XB <- rnorm(N,0,sqrt(s2vec))
  DatB <- data.frame(XB=XB,grp=group)
  
  # Calculate means and variances
  XbarB <- aggregate(XB~group,data=DatB,mean)[,2]
  si2B <-aggregate(XB~group,data=DatB,var)[,2]
  ShatB <- diag(si2B/n)
  GammahatB <- C%*%ShatB%*%t(C)
  diffB <- C%*%XbarB
  
  # Test statistic
  param_TlB[,h] <- diffB/sqrt(c(diag(GammahatB)))
  param_Al[h] <- max(abs(param_TlB[,h]))
}

# pvalue
param_pvalue = mean(param_Al>=T0)

cat("non-parametric : ", non_pvalue, ", parametric : ", param_pvalue)

# non-parametric :  0 , parametric :  0

