import math
import numpy as np

def shannon_entropy(list_input):
    """Calculate Shannon's Entropy per column of the alignment (H=-\sum_{i=1}^{M} P_i\,log_2\,P_i)"""

    unique_base = set(list_input)
    M = len(list_input)
    entropy_list = []
    # Number of residues in column
    for base in unique_base:
        n_i = list_input.count(base) # Number of residues of type i
        P_i = n_i/float(M) # n_i(Number of residues of type i) / M(Number of residues in column)
        entropy_i = P_i*(math.log(1/P_i,2))
        entropy_list.append(entropy_i)

    sh_entropy = sum(entropy_list)

    return sh_entropy

def entropy_of_each_col(df):
    shannon_entropy_list = []
    for i in range(len(df.columns)):
        if any(df[i].astype('string')=='-') or any(df[i].astype('string')=='N'):
            shannon_entropy_list.append(int(-1))
        else:
            list_input = df[i].tolist()
            shannon_entropy_list.append(shannon_entropy(list_input))

    return shannon_entropy_list

def avg_variability(shannon_entropy_list, window_size):
    avg_variability=[]
    for i in range(int(len(shannon_entropy_list))):
        if i >= len(shannon_entropy_list) - window_size:
            break
        start = i
        end = start + window_size
        number_list = shannon_entropy_list[start:end]
        avg = np.mean(number_list)
        avg_variability.append(avg)

    return avg_variability
