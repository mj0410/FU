from Bio import AlignIO
from func_variability import shannon_entropy, entropy_of_each_col, avg_variability
import pandas as pd
import numpy as np

#input parameter
input_data = str(snakemake.input)
window_size = 100

#read data
align = AlignIO.read(input_data, "fasta")

align_df = pd.DataFrame(align)

shannon_entropy = entropy_of_each_col(align_df)

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
plt.plot(shannon_entropy)
plt.ylabel("variability")
plt.xlabel("genome position")
plt.savefig(str(snakemake.output.plot))

avg_var = avg_variability(shannon_entropy, window_size)
avg_df = pd.DataFrame(avg_var)
np.savetxt(str(snakemake.output.txt), avg_df.values, fmt='%1.3f')
