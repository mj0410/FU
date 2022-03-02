'''
Module used to extract features,
Only extract columns that has a valid header from ./headers/{}_header_valid.csv
'''

import os
import sys
BASEPATH = os.environ['BASEPATH']
sys.path.append(BASEPATH)
from os.path import join
import argparse
import pandas as pd
import time
import itertools
from multiprocessing import Pool
from tqdm import tqdm
import functools

from helpers.read_raw_data import *
from utils import get_valid_types, str_or_none, str2bool
from helpers.utils import valid_header_iter_gen, count_length_gen

# Get rid of gensim deprecated warning
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

TYPENAME = os.environ['TYPENAME']
INPUT_DIR = os.environ['INPUT_DIR']
valid_types = get_valid_types(TYPENAME)
valid_header_dir = os.path.join(os.environ['BASEPATH'], 'extract', 'out', 'headers')


if __name__ == "__main__": 


    MAX_FIELDS = 10000
    cache_size = 100

    # Get corpus
    parser = argparse.ArgumentParser()
    parser.add_argument('header_file', type=str)
    parser.add_argument('-O', '--output', type=str, default='feature_extraction')
    parser.add_argument('-f', '--features', type=str, nargs='?', default='sherlock', choices=['sherlock', 'topic'])
    parser.add_argument('-LDA', '--LDA_name', nargs='?', type=str_or_none, default=None)
    parser.add_argument('-n', '--num_processes', nargs='?', type=int, default=4)
    parser.add_argument('-o', '--overwrite', nargs='?', type=str2bool, default=False)
    parser.add_argument('-c', '--config', type=str, default=None)

    args = parser.parse_args()
    header_file = args.header_file
    output_name = args.output
    config = args.config

    # Create features directory
    features_dir = join(BASEPATH, 'extract', 'out', 'features')
    if not os.path.exists(features_dir):
        os.makedirs(features_dir)

    if args.features == 'topic':
        print("TOPIC MODEL")
        assert args.LDA_name is not None, "Must provide an LDA_name"

        os.environ['LDA_name'] = args.LDA_name
        # hack to pass in LDA name for extractor
        from feature_extraction.topic_features_LDA import extract_topic_features
        from gensim.corpora.dictionary import Dictionary
        from gensim.models.ldamodel import LdaModel

        feature_name = "{}_{}".format(args.features, args.LDA_name)
        extract_func = extract_topic_features

    elif args.features == 'sherlock':
        from feature_extraction.sherlock_features import *
        feature_name = args.features
        if config is None:
            extract_func = extract_sherlock_features
        else:
            extract_func = extract_sherlock_features_pred
    else:
        print('Invalid feature names')
        exit(1)


    print('Extracting features for feature group {}'.format(args.features))

    output_file = join(features_dir, '{}_{}_{}.parquet'.format(output_name, TYPENAME, feature_name))
    if not args.overwrite:
        assert not os.path.isfile(output_file), "\n {} already exists".format(output_file)

    header_dir = join(BASEPATH, 'extract/out/headers', header_file)
    #header_iter = valid_header_iter_gen(header_name) # Iterator for reading large header file.

    raw_df_iter = get_filtered_dfs(header_dir)
    header_length = count_length_gen(header_dir)
    print("header length : {}".format(header_length))

    ########################################
    # Distribute the tasks using pools
    ########################################
    start_time = time.time()
    task_pool = Pool(args.num_processes)

    cache = []
    counter = 0
    for df_features in tqdm(task_pool.imap(extract_func, raw_df_iter), total=header_length):
       counter += 1
       cache.append(df_features)
       #print("cache : ", cache)

    print("counter : {}".format(counter))
    df = pd.concat(cache)
    df['row_index'] = df['row_index'].map(str)
    df.to_parquet(output_file, compression='brotli', index=False) #header=True, index=False,

    task_pool.close()
    task_pool.join()

    end_time = time.time()
    print("Feature extraction done ({} sec)".format(int(end_time - start_time)))
    
'''
    counter = 0
    header, mode = True, 'w'
    col_counter = 0
    cache = []
    #, desc='{} processes'.format(args.num_processes)

    for df_features in tqdm(task_pool.imap(extract_func, raw_df_iter), total=header_length):
        counter += 1
        cache.append(df_features)
        if counter % cache_size == 0:
            df = pd.concat(cache)
            df.to_csv(output_file, header=header, index=False, mode=mode)
            col_counter += len(df)
            header, mode = False, 'a'
            cache = []

    #save the last cache
    if len(cache) > 0:
        df = pd.concat(cache)
        df.to_csv(output_file, header=header, index=False, mode=mode)

    print("Number of columns: {}".format(col_counter))
    
    task_pool.close()
    task_pool.join()
'''
