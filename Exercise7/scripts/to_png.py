import ete3
from ete3 import Tree

input = str(snakemake.input)
output = str(snakemake.output)

t = Tree(input)
t.render(output, dpi=200)
