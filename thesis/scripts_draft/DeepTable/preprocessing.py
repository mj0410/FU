import pickle
import pandas as pd
import numpy as np
import os
from os.path import join
import random
from collections import Counter

def csv_to_pkl(path, inp, file_name, pred=None, MAX_COL=None, MAX_COL_LENGTH=None):

  table_list, label, tables = [], [], []

  for i in inp:
    df = pd.read_csv(join(path, i))
    if MAX_COL != None:
      df = df.iloc[:MAX_COL, :MAX_COL_LENGTH]
    df = df.T.reset_index().values.T.tolist()
    df_str = [list(map(str, x)) for x in df]
    table_list.append(df_str)

  if pred is None:
    for i in range(len(table_list)):
      label.append(np.array([1, 0, 0]))
  else:
    label = inp

  tables.append(table_list)
  tables.append(label)

  pickle.dump(tables, open(file_name+".pkl", "wb"))
  pkl_file = str(file_name)+".pkl"

  return pkl_file

def split_to_pkl(input_file, output_name, pred=None):

  dataset_id, row_index, content, label, tables = [], [], [], [], []
  table_names = input_file.dataset_id.unique()

  for id in table_names:
    df_id = input_file[input_file['dataset_id']==id]
    file_path = df_id.iloc[0]['locator']
    df = pd.read_csv(join(file_path, id))

    for i in df_id['row_index'].values:
      df_rows = df.loc[eval(i)]
      df_rows = df_rows.T.reset_index().values.T.tolist()
      df_str = [list(map(str, x)) for x in df_rows]
      content.append(df_str)
      dataset_id.append(id)
      row_index.append(i)

  for i in range(len(content)):
    label.append(np.array([1, 0, 0]))

  tables.append(dataset_id)
  tables.append(row_index)
  tables.append(content)
  tables.append(label)

  pickle.dump(tables, open(output_name+".pkl", "wb"))
  pkl_file = str(output_name)+".pkl"

  return pkl_file

def split_to_random_pkl(input_file, output_name, pred=None):

  dataset_id, row_index, content, label, tables = [], [], [], [], []
  table_names = input_file.dataset_id.unique()

  for id in table_names:
    df_id = input_file[input_file['dataset_id']==id]
    file_path = df_id.iloc[0]['locator']
    df = pd.read_csv(join(file_path, id))

    rand_list = [random.randint(0, 1) for _ in range(len(df_id))]
    print("count 1 and 0 ", Counter(rand_list))

    for i in range(len(df_id)):
      row_idx = df_id['row_index'].values[i]
      df_rows = df.loc[eval(row_idx)]
      if rand_list[i] == 0:
        df_rows = df_rows.T.reset_index().values.T.tolist()
        label.append(np.array([1, 0, 0]))
      else:
        df_rows = df_rows.T.reset_index().values.tolist()
        label.append(np.array([0, 1, 0]))
      df_str = [list(map(str, x)) for x in df_rows]

      content.append(df_str)
      dataset_id.append(id)
      row_index.append(i)

  tables.append(dataset_id)
  tables.append(row_index)
  tables.append(content)
  tables.append(label)

  pickle.dump(tables, open(output_name+".pkl", "wb"))
  pkl_file = str(output_name)+".pkl"

  return pkl_file
