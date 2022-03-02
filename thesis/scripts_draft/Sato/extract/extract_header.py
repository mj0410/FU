'''
Extract headers,
Filter header before generating feature vectors
Format of output_name.csv:
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Columns:        Descriptions
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Locator         The path of the datafile
dataset_id      The dataset of the table. one file has multiple tables except manyeyes
field_list      The list of field whose header is a valid type. numbers are index in table
field_names     The list of field names. numbers are index in the corresponsing type list

'''

import os
import sys

BASEPATH = os.environ['BASEPATH']
sys.path.append(BASEPATH)

from os.path import join
from os import listdir
import argparse
import pandas as pd
import random, math
from collections import OrderedDict
from helpers.utils import canonical_header
from utils import get_valid_types, str2bool

from helpers.read_raw_data import get_dfs

# Get rid of gensim deprecated warning
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

raw_data_dir = os.environ['INPUT_DIR']
TYPENAME = os.environ['TYPENAME']
valid_types = get_valid_types(TYPENAME)
print("valid type length : {}".format(len(valid_types)))

# field_list : 주어진 df의 col 중 주어진 type list의 type들과 일치하는? 애들의 col번호
# field_names : type list에서의 각 valid col type의 번호

def get_valid_headers(df_iter, num):
    for df_dic in df_iter:
        df, locator, dataset_id = df_dic['df'], df_dic['locator'], df_dic['dataset_id']
        idx_list = list(df.index)
        random.shuffle(idx_list)
        row_num = num

        if row_num > len(idx_list):
            row_num = len(idx_list)
        if row_num == 0:
            row_num = 1
        chunks = [idx_list[x:x + row_num] for x in range(0, len(idx_list), row_num)]

        for idx in chunks:
            valid_fields = []
            field_names = []  # index of the headers, according to the type used.
            row_index = idx
            df_tenrows = df.loc[idx]
            for field_order, field_name in enumerate(df_tenrows.columns):

                # canonicalize headers
                field_name_c = canonical_header(field_name)

                # filter the column name
                if field_name_c in valid_types:
                    valid_fields.append(field_order)
                    field_names.append(valid_types.index(field_name_c))

            if len(valid_fields) > 0:
                table_valid_headers = OrderedDict()

                table_valid_headers['locator'] = locator
                table_valid_headers['dataset_id'] = dataset_id
                table_valid_headers['field_list'] = valid_fields
                table_valid_headers['field_names'] = field_names
                table_valid_headers['row_index'] = row_index
                yield table_valid_headers

def get_valid_headers_pred(df_iter):
    for df_dic in df_iter:
        df, locator, dataset_id = df_dic['df'], df_dic['locator'], df_dic['dataset_id']

        valid_fields = []
        field_names = []  # index of the headers, according to the type used.
        for field_order, field_name in enumerate(df.columns):

            # canonicalize headers
            field_name_c = canonical_header(field_name)

            # filter the column name
            if field_name_c in valid_types:
                valid_fields.append(field_order)
                field_names.append(valid_types.index(field_name_c))

        if len(valid_fields) > 0:
            table_valid_headers = OrderedDict()

            table_valid_headers['locator'] = locator
            table_valid_headers['dataset_id'] = dataset_id
            table_valid_headers['field_list'] = valid_fields
            table_valid_headers['field_names'] = field_names
            yield table_valid_headers


parser = argparse.ArgumentParser()
parser.add_argument('output_file', type=str, help='name of output file')
parser.add_argument('-n', '--table_num', type=int, default=10)
parser.add_argument('-o', '--overwrite', nargs='?', type=str2bool, default=False)
parser.add_argument('-c', '--config', type=str, default=None)

args = parser.parse_args()
header_path = join(os.environ['BASEPATH'], 'extract', 'out', 'headers')
header_name = args.output_file
header_file = join(header_path, header_name)

table_num = args.table_num
config = args.config

print("header_path = {}".format(header_path))
if not os.path.exists(header_path):
    os.makedirs(header_path)

df_iter = get_dfs(raw_data_dir)
if config is None:
    header_iter = get_valid_headers(df_iter, table_num)
else:
    header_iter = get_valid_headers_pred(df_iter)

if not args.overwrite:
    assert not os.path.isfile(header_file), "\n \n {} already exists".format(header_file)

print("save {}".format(header_name))
df = pd.DataFrame(header_iter)
print("number of tables : ", len(df))
df.to_pickle(header_file)

