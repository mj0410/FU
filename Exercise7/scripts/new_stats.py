import pandas as pd
import numpy as np

df=pd.read_csv((snakemake.input[0]), header=None, delim_whitespace=True)
df['depth'] = df[2]/df[1]

df.to_csv(snakemake.output[0], header=True, index=False, sep=',', mode='a')
