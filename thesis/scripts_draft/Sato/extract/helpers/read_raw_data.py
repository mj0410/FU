'''
Helper functions to read through raw data for each corpus. 
Each is a generator that yields a single dataset and metadata in the form:

{
    'df'
    'locator',
    'dataset_id'
}
'''
import os
from os import listdir
from os.path import join
from collections import OrderedDict
import argparse
import gzip
import json
import chardet
import traceback
import itertools
import numpy as np
import pandas as pd

def load_csvs(file_path, id, valid_fields=None, num_row=None):
    file_name = join(file_path, id)
    df = pd.read_csv(file_name, keep_default_na=False)

    if num_row is not None:
        df = df.loc[num_row]

    if valid_fields is not None:
        df = df.iloc[:, valid_fields]

    result = {
      'df': df,
      'dataset_id': id,
      'locator': file_path
      }
    return result


def get_dfs(raw_data_dir):
    files = [f for f in listdir(raw_data_dir) if f.endswith('.csv')]
    print("number of files : {}".format(len(files)))

    for file_name in files:
        locator = raw_data_dir
        dataset_id = file_name

        df = load_csvs(locator, dataset_id)
        if df:
            yield df


def get_filtered_dfs(full_header_file_dir):
    header = pd.read_pickle(full_header_file_dir)
    #header = header[:10]
    for line in header.iterrows():
        #print(line)
        idx, row = line
        locator = row['locator']
        dataset_id = row['dataset_id']
        fields = row['field_list']  # convert string to list : eval()
        num_row = row['row_index']

        df = load_csvs(locator, dataset_id, fields, num_row)
        df['row_index'] = num_row
        if df:
            yield df

