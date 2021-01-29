### Construct Phylogenetic network in R

```bash
library(seqinr)
library(phangorn)

align <- read.alignment(file="aligned.fasta", format="fasta")
distance <- dist.alignment(align, matrix="identity")

nnet <- neighborNet(distance)
plot(nnet, "2D")
```