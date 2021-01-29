import ete3
from ete3 import Tree

import os

os.environ['QT_QPA_PLATFORM']='offscreen'

input = str(snakemake.input)
output = str(snakemake.output)

t = Tree(input)
t.render(output, dpi=200)
