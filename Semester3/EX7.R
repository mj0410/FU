##### Exercise 1 #####

# Poisson distribution
# lambda=300 per hour / 5 per minute

# 1. Compute the probability that none passes in a given minute

# Calculate P(X=0)
prob = exp(-5)

# 2. What is the expected number of passing vehicles each two munutes?

# 10 since the average number of passing vehicle is 300 per hour

# 3. Probability that the expected number actually pass through in a given two-minute period

# lambda=10 in this case
# Calculate P(X=10) with lambda 10
prob = exp(-10)*10^10/factorial(10)



##### Exercise 2 #####

# Negative binomial distribution
# probability of success = 20%
p=0.2

# 1. What is the probability that the first strike comes on the third well drilled?

# First and second try should be failed -> 80% for each
# Third try should be success -> 20%
r=1; k=2
prob = (p^r)*((1-p)^k)

# 2. What is the probability that the third strike comes on the seventh well drilled?

# Calculate P(X=4) in the slide 11 of class 8
r=3; k=4
prob = choose(6,4)*(p^r)*((1-p)^k)

# 3. What is the mean and variance of the number of wells that must be drilled 
# if the oil company wants to set up three producing wells?

# Calculate E(X)=r(1-p)/p, Var(X)=r(1-p)/p^2
r=3
mx=r(1-p)/p
vx=r(1-p)/(p^2)



##### Exercise 3 #####

female=c(4, 3, 4, 6, 2, 5, 2, 4, 1, 3, 4, 2, 2, 5, 3)
male=c(6, 5, 2, 2, 5, 4, 3, 3, 5, 7, 5, 6, 2, 5, 2)

# 1. Compute means and variances
mf=mean(female)
vf=var(female)

mm=mean(male)
vm=var(female)

# 2. Poisson fit
pf<-exp(-mf)*mf^female/factorial(female)
pm<-exp(-mm)*mm^male/factorial(male)

par(mfrow=c(2,2))
barplot(height = table(factor(female, levels=min(female):max(female)))/length(female),
        ylab = "proportion", main="Female")
barplot(height = table(factor(male, levels=min(male):max(male)))/length(male),
        ylab = "proportion", main="Male")
barplot(height = table(factor(pf))/length(pf), names.arg=c(1,2,3,4,5,6),
        ylab = "poisson fit", main="Poisson fit (Female)")
barplot(height = table(factor(pm))/length(pm), names.arg=c(2,3,4,5,6,7),
        ylab = "poisson fit", main="Poisson fit (Male)")

# The given data don't have poisson distribution


# 3. Test H0:log(mu1/mu2)=0
les=data.frame(res=c(female,male), grp=factor(c(rep(1,15),rep(2,15))))

#poisson
fitPois<-glm(res~grp, family="poisson", data=les)
summary(fitPois)
estPois<-cbind(Estimate=coef(fitPois), confint(fitPois))
SEPois=coef(summary(fitPois))[,2][2]

#negative binomial
library(MASS)
fit<-glm.nb(res~grp, data=les)
summary(fit)
SENB<-coef(summary(fit))[,2][2]
est<-cbind(Estimate=coef(fit), confint(fit))


# 4. Compute the estimated treatment effect and standard errors

# estimated treatment effect = log(mean(male)/mean(female))
log(mm/mf)

# standard error = sd/sqrt(number of sample)
SEf=sqrt(vf)/sqrt(length(female))
SEm=sqrt(vm)/sqrt(length(male))


# 5. Perform type 1 error simulation

simulation<-function(l1,l2,n1,n2,nsim){
  erg=c()
  
  for(i in 1:nsim){
    #generate poisson distribution data
    x=rpois(n1,l1)
    y=rpois(n2,l2)
  
    les=data.frame(res=c(x,y), grp=factor(c(rep(1,n1),rep(2,n2))))
  
    fitPois<-glm(res~grp, family="poisson", data=les)
    p=coef(summary(fitPois))[,4][2]
    erg[i]=(p<0.05)
  }
  
  result<-data.frame(n1=n1, n2=n2, Pois=mean(erg))
  print(result)
}

n1=n2=c(10,15,20,50)

for(i in n1){
  for(j in n2){
    simulation(3,3,i,j,1000)
  }
}

# n1 n2  Pois
# 10 10 0.043
# 10 15 0.04
# 10 20 0.059
# 10 50 0.036
# 15 10 0.043
# 15 15 0.047
# 15 20 0.06
# 15 50 0.048
# 20 10 0.047
# 20 15 0.05
# 20 20 0.04
# 20 50 0.06
# 50 10 0.054
# 50 15 0.048
# 50 20 0.05
# 50 50 0.049

# The poisson regression method can control type 1 error rate.