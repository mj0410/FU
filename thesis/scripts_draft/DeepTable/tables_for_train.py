
import pandas as pd
import numpy as np
import pickle
import os
from os import listdir
from os.path import join


def pkl_for_train(inp, input_file, output_file):

	MAX_COL = 9
	MAX_COL_LENGTH = 9

	with open(inp, 'rb') as f:
		[X, y] = pickle.load(f)

	before = len(X)

	tables = []
	table_names = input_file.dataset_id.unique()

	for id in table_names:
		print(id)
		df_id = input_file[input_file['dataset_id'] == id]
		file_path = df_id.iloc[0]['locator']
		df = pd.read_csv(join(file_path, id))

		for i in df_id['row_index'].values:
			df_rows = df.loc[eval(i)]
			df_rows = df_rows.iloc[:MAX_COL, :MAX_COL_LENGTH]
			df_rows = df_rows.T.reset_index().values.T.tolist()
			df_str = [list(map(str, x)) for x in df_rows]
			X.append(df_str)

	after = len(X)
	for i in range(after-before):
		y.append(np.array([1, 0, 0]))

	tables.append(X)
	tables.append(y)
	print("table length : ", len(X), len(y))

	pickle.dump(tables, open(output_file + ".pickle", "wb"))
	print("pkl saved")
	
if __name__ == "__main__":

	SPLIT = os.environ['SPLIT']
	inp = os.environ['INPUT_TABLES']

	input_file = pd.read_csv(SPLIT)

	pkl_for_train(inp, input_file, 'tables_10p')