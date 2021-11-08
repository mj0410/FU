library(phangorn)
library(seqinr)

align <- read.alignment(file="C:/Users/minie/Desktop/aligned.fasta", format="fasta")
distance <- dist.alignment(align, matrix="identity")

nnet <- neighborNet(distance)
test <- nnet

data<-read.csv("C:/Users/minie/Downloads/sequences.csv", header=T)

asia <- c()
africa <- c()
namerica <- c()
samerica <- c()
europe <- c()
oce <- c()


for(i in 1:137){
  for(j in 1:137){
    if(test$tip.label[i] == data$Accession[j]){
      if(data$continent[j]=='Asia'){
        asia <- append(asia, i)
      }
      if(data$continent[j]=='Africa'){
        africa <- append(africa, i)
      }
      if(data$continent[j]=='North America'){
        namerica <- append(namerica, i)
      }
      if(data$continent[j]=='South America'){
        samerica <- append(samerica, i)
      }
      if(data$continent[j]=='Europe'){
        europe <- append(europe, i)
      }
      if(data$continent[j]=='Ocenia'){
        oce <- append(oce, i)
      }
    }
  }
}

# find edges that are in the network but not in the tree
edge.col <- rep("grey", nrow(test$edge))
edges_to_leaf <- c()

for(i in 1:length(test$edge[,1])){
  if(test$edge[i,2] %in% asia){
    edge.col[i] <- 'pink'
    edges_to_leaf <- append(edges_to_leaf, i)
    print('asia')
  }else if(test$edge[i,2] %in% africa){
    edge.col[i] <- 'orange'
    edges_to_leaf <- append(edges_to_leaf, i)
    print('africa')
  }else if(test$edge[i,2] %in% namerica){
    edge.col[i] <- 'yellow'
    edges_to_leaf <- append(edges_to_leaf, i)
    print('north america')
  }else if(test$edge[i,2] %in% samerica){
    edge.col[i] <- 'green'
    edges_to_leaf <- append(edges_to_leaf, i)
    print('south america')
  }else if(test$edge[i,2] %in% europe){
    edge.col[i] <- 'skyblue'
    edges_to_leaf <- append(edges_to_leaf, i)
    print('europe')
  }else if(test$edge[i,2] %in% oce){
    edge.col[i] <- 'purple'
    edges_to_leaf <- append(edges_to_leaf, i)
    print('ocenia')
  }
}

plot(test, "2D", edge.color=edge.col, show.tip.label = FALSE)

nodes_before_leaves <- c()
nodes <- c()

for(i in 1:length(test$edge[,1])){
  if(i %in% edges_to_leaf){
    nodes_before_leaves <- append(nodes_before_leaves, test$edge[i,1])
  }
}

for(i in 1:length(test$edge[,1])){
  if(test$edge[i, 2] %in% nodes_before_leaves){
    nodes <- append(nodes, test$edge[i,1])
  }
}
