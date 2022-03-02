import numpy as np
import random 
import pickle
import os
os.environ['KERAS_BACKEND']='tensorflow'
import keras
from keras.preprocessing.text import Tokenizer, text_to_word_sequence

def transform_tables(inp, MAX_COL, MAX_COL_LENGTH, MAX_CELL_LENGTH, config=None):

	with open(inp, 'rb') as f:
		[X, y] = pickle.load(f)

	#print(X)

	texts = ["XXX"] + [' '.join(text_to_word_sequence(' '.join(sum(x,[])),lower=True)) for x in X]

	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(texts)

	data = np.zeros((len(X), MAX_COL, MAX_COL_LENGTH, MAX_CELL_LENGTH), dtype='int32')

	X = X[0:len(X)]
	y = y[0:len(X)]

	for i, table in enumerate(X):
		for j, col in enumerate(table):
			if j < MAX_COL:
				for k, cell in enumerate(col):
					if k < MAX_COL_LENGTH:
						z = 0
						for _, word in enumerate(text_to_word_sequence(cell,lower=True)):
							if z<MAX_CELL_LENGTH:
								if tokenizer.word_index.get(word) is not None:
									data[i, j, k, z] = tokenizer.word_index[word]
									z = z+1

	if config is None:
		y = np.array(y)

	return data, y, tokenizer.word_index

def transform_split_tables(inp, MAX_COL, MAX_COL_LENGTH, MAX_CELL_LENGTH, config=None):

	print("Input transformation...")

	with open(inp, 'rb') as f:
		[table_name, row_index, X, y] = pickle.load(f)

	texts = ["XXX"] + [' '.join(text_to_word_sequence(' '.join(sum(x,[])),lower=True)) for x in X]

	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(texts)

	data = np.zeros((len(X), MAX_COL, MAX_COL_LENGTH, MAX_CELL_LENGTH), dtype='int32')

	X = X[0:len(X)]
	y = y[0:len(X)]
	y = np.array(y)

	for i, table in enumerate(X):
		for j, col in enumerate(table):
			if j < MAX_COL:
				for k, cell in enumerate(col):
					if k < MAX_COL_LENGTH:
						z = 0
						for _, word in enumerate(text_to_word_sequence(cell,lower=True)):
							if z<MAX_CELL_LENGTH:
								if tokenizer.word_index.get(word) is not None:
									data[i, j, k, z] = tokenizer.word_index[word]
									z = z+1

	return table_name, row_index, data, y, tokenizer.word_index
	
