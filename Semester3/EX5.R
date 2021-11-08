mypermu<-function(nsim,nperm,n1,n2,s1,s2){
  N<-n1+n2;erg<-c()
  #-----------Permutation Matrices--------------#
  i1<-c(rep(1,n1),rep(-1,n2))
  P<-t(apply(matrix(i1,nrow=nperm,ncol=N,byrow=TRUE),1,sample))
  
  set.seed(1)
  x1<-matrix(rnorm(n1*nsim)*sqrt(s1),ncol=nsim,nrow=n1)
  x2<-matrix(rnorm(n2*nsim)*sqrt(s2),ncol=nsim,nrow=n2)
  
  x<-rbind(x1,x2)
  mx1<-colMeans(x1); mx2<-colMeans(x2)
  Tns <-(mx1-mx2)/sqrt((s1/n1)+(s2/n2))
  

  for (i in 1:nsim){
    TnsP<-c()
    #---------------Permutations---------------#
    for(k in 1:nperm){
      x1star=c()
      x2star=c()
      
      for(j in 1:N){
        if(P[k,j]==1){
          x1star <- append(x1star, x[j,i])
        }
        else{
          x2star <- append(x2star, x[j,i])
        }
      }
    
      mx1star = mean(x1star)
      v1star = var(x1star)
      mx2star = mean(x2star)
      v2star = var(x2star)
      
      TnsP[k] = (mx1star-mx2star)/sqrt((v1star/n1)+(v2star/n2))
    }
    pvalue<-2*min(mean(TnsP<=Tns[i]),mean(TnsP>=Tns[i]))
    erg[i] <-(pvalue<0.05)
  }
  
  result<-data.frame(nsim=nsim,nperm=nperm,n1=n1,n2=n2,s1=s1,s2=s2,
                     Permu=mean(erg))
  result
}

mypermu(1000,1000,10,10,1,1)
mypermu(1000,1000,10,10,1,3)

mypermu(1000,1000,10,20,1,1)
mypermu(1000,1000,10,20,1,3)

mypermu(1000,1000,20,10,1,1)
mypermu(1000,1000,20,10,1,3)

mypermu(1000,1000,50,100,1,1)
mypermu(1000,1000,50,100,1,3)

mypermu(1000,1000,100,50,1,1)
mypermu(1000,1000,100,50,1,3)

#mypermu(10000,10000,10,10,1,1)
#mypermu(10000,10000,10,10,1,3)

#mypermu(10000,10000,10,20,1,1)
#mypermu(10000,10000,10,20,1,3)

#mypermu(10000,10000,20,10,1,1)
#mypermu(10000,10000,20,10,1,3)

#mypermu(10000,10000,50,100,1,1)
#mypermu(10000,10000,50,100,1,3)
