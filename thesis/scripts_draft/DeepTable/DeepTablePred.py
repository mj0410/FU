
import pandas as pd
import numpy as np
import random
import os
#os.environ['KERAS_BACKEND']='tensorflow'
import keras
from keras.models import Model,Sequential,load_model
#np.random.seed(813306)
from input_transformation import *
from preprocessing import *
import configargparse
from os import listdir
	
if __name__ == "__main__":

	p = configargparse.ArgParser()
	p.add('-m', '--model', type=str, required=True, help='trained model')
	p.add('-i', '--input_path', type=str, help='input tables path')
	p.add('-o', '--output', type=str, default='result', help='output file name')

	args = p.parse_args()

	trained_model = args.model
	inp = args.input_path
	output = args.output
	
	# variable initialization
	MAX_COL=9
	MAX_COL_LENGTH=9
	MAX_CELL_LENGTH=4

	input_list = listdir(inp)

	preprocessed = csv_to_pkl(inp, input_list, 'preprocessed', 'pred', MAX_COL, MAX_COL_LENGTH)
	X_test, table_id, dictionary = transform_tables(preprocessed, MAX_COL, MAX_COL_LENGTH, MAX_CELL_LENGTH, config='pred')
	print("\ntable dictionary : ", len(dictionary))

	# load model
	model = load_model(trained_model)

	# predict labels
	print("prediction..")

	pred = model.predict(X_test, verbose=1)
	preds = [p.tolist().index(max(p.tolist())) for p in pred]
	locator = [inp]*len(preds)
	
	# write predictions in a file
	data_dict = {'locator':locator, 'table':table_id, 'prediction':preds}
	pred_result = pd.DataFrame(data_dict)
	pred_result.to_csv(join(os.environ['ORIGIN'], "{}.csv".format(output)), index=False)

	print("predictions are saved in \""+output+".csv\"")
	#print(pred_result.prediction.sum())
