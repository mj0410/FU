import pandas as pd
import os, sys
BASEPATH = os.environ['BASEPATH']
sys.path.append(BASEPATH)
from os import listdir
from os.path import join
import configargparse
import json
from helpers.utils import canonical_header2, reduce_header, reduce_header2

def read_input(inp):
  input_list = []
  for file in [f for f in listdir(inp) if f.endswith('.csv')]:
    input_list.append(file)
  return input_list

def read_header(inp):
  table = pd.read_csv(inp, keep_default_na=False)
  #table.dropna(how='all', axis=1, inplace=True)
  col_list = list(table.columns.values)
  return col_list

p = configargparse.ArgParser()
p.add('--file_path', required=True, type=str, help='table files path')
p.add('--type_name', required=True, type=str, help='name of saved type')

args = p.parse_args()
types_file = join(BASEPATH, 'configs', 'types.json')
files_path = args.file_path
type_name = args.type_name

# extract column names of input tables
file_list = read_input(files_path)
print("file_list : ", file_list)
header_list = []
for file in file_list:
  col_list = read_header(join(files_path, file))
  header_list = header_list + col_list
print("number of col names before remove dups : ", len(header_list))
header_list = list(map(canonical_header2, header_list))
header_list = list(dict.fromkeys(header_list))
print(header_list)
print("number of types : ", len(header_list))

# update types.json file
types = open(types_file, "r")
types_dict = json.load(types)
types.close()

types_dict.update({type_name: header_list})
print(types_dict.keys())

types = open(types_file, "w")
json.dump(types_dict, types)
types.close()