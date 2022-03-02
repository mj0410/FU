
import pandas as pd
import numpy as np
import random 
import pickle
import os
os.environ['KERAS_BACKEND']='tensorflow'
from os import listdir
import keras
from keras.layers import Embedding,Dense, Input, Flatten,Conv1D,Conv2D, MaxPooling1D, Embedding, Concatenate, Dropout,AveragePooling1D,LSTM, GRU, Bidirectional, TimeDistributed,Convolution2D,MaxPooling2D,AveragePooling2D,Permute
from keras.layers.core import Permute
from keras.models import Model,Sequential,load_model 
from sklearn.metrics import classification_report as cr
from sklearn.metrics import confusion_matrix as cm
from keras import backend as K
np.random.seed(813306)	
from input_transformation import *
from preprocessing import *
import configargparse

	
if __name__ == "__main__":

	BASEPATH = os.environ['BASEPATH']

	p = configargparse.ArgParser()
	p.add('-m', '--model', type=str, required=True, help='trained model')
	p.add('-ip', '--input_path', type=str, default=None, help='input tables path')
	p.add('-if', '--input_file', type=str, default=None, help='test.pkl')
	p.add('-o', '--output', type=str, default='result', help='output file name')

	args = p.parse_args()

	trained_model = args.model
	inp_path = args.input_path
	inp = args.input_file
	output = args.output
	
	# variable initialization
	MAX_COL=9
	MAX_COL_LENGTH=9
	MAX_CELL_LENGTH=4

	if inp_path is not None:
		input_list = listdir(inp_path)
		print("input_list : ", input_list)
		preprocessed = csv_to_pkl(inp_path, input_list, 'test')
		X_test, y_test, dictionary = transform_tables(preprocessed)
	else:
		with open(inp, 'rb') as f:
			[X_test, y_test] = pickle.load(f)
	
	# load model
	model = load_model(trained_model)
	
	# predict labels
	pred = model.predict(X_test, verbose=1)

	refs = [r.tolist().index(max(r.tolist())) for r in y_test]
	preds = [p.tolist().index(max(p.tolist())) for p in pred]
	
	# write predictions in a file
	refs_preds = pd.DataFrame([(r.tolist().index(max(r.tolist())),p.tolist().index(max(p.tolist()))) for r,p in zip(y_test,pred)], columns = ["reference","prediction"])
	refs_preds.to_csv(output+".csv",index=False)
	
	# display performances
	print(cr(refs, preds, digits = 4))
	print("confusion_matrix:\n", cm(refs, preds))
	print("predictions are saved in \""+output+".csv\"")