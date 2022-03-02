import pandas as pd
import os, sys
BASEPATH = os.environ['BASEPATH']
sys.path.append(BASEPATH)
import numpy as np
from os.path import join
from sklearn.model_selection import train_test_split
import json
import configargparse
import pickle
from utils import int_or_none

TYPENAME = os.environ['TYPENAME']
header_path = join(os.environ['BASEPATH'], 'extract/out/headers')

tmp_path = 'out/train_test_split'
if not os.path.exists(tmp_path):
    os.makedirs(tmp_path)

p = configargparse.ArgParser()
#p.add('--mode', type='str', choices=['split', 'prediction'], help='split : split train/validation/test data | prediction : prepare list of data for prediction')
p.add('--header_file', type=str, help='name of extracted header file')
p.add('--output', type=str, help='name of output file')
p.add('--val_percent', type=int_or_none, default=None, help='percentage for validation set')
p.add('--test_percent', type=int, default=20, help='percentage for test set')
p.add('--cv', type=int_or_none, default=None, help='number of cross validation')

args = p.parse_args()
header_file_name = args.header_file
output_file = args.output
val_p = args.val_percent
test_p = args.test_percent
cv = args.cv

header_file = join(header_path, header_file_name)
df = pd.read_pickle(header_file)

print('Spliting {}'.format(header_file_name))

if cv is not None:
    train_df, test_df = train_test_split(df, test_size=test_p * 0.01, random_state=42)

    shuffled = train_df.sample(frac=1)
    split_dfs = np.array_split(shuffled, cv)
    cross_validation = [0] * len(df)

    for j in range(cv):
        for i in range(len(train_df)):
            if train_df.index[i] in split_dfs[j].index:
                cross_validation[train_df.index[i]] = j + 1
    for k in range(len(test_df)):
        cross_validation[test_df.index[k]] = 'test'

    df['cv'] = cross_validation
    print("Done for {}-fold cross validation".format(cv))
    dataset_path = join(tmp_path, '{}_{}fold_{}.csv'.format(output_file, cv, TYPENAME))

else:
    train_df, test_df = train_test_split(df, test_size=test_p*0.01, random_state=42)
    train_df, val_df = train_test_split(train_df, test_size=val_p*0.01, random_state=42)

    df['split'] = df.index.map(lambda x: 'train' if x in list(train_df.index) else ('test' if x in list(test_df.index) else 'val'))
    print("Done, {} training tables, {} validation tables, {} testing tables".format(len(train_df), len(val_df), len(test_df)))
    dataset_path = join(tmp_path, '{}_{}.csv'.format(output_file, TYPENAME))

df.to_csv(dataset_path)
print("csv saved")
