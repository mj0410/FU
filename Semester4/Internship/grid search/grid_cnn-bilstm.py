# -*- coding: utf-8 -*-

import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import tarfile
import pandas as pd
import numpy as np
import re
from datetime import datetime

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dropout, Activation, Flatten, Dense
from keras.layers import LSTM, Bidirectional, Conv1D, MaxPooling1D

from matplotlib import pyplot as plt
from Bio.Seq import Seq

"""### Read data"""

filename = "training_data_v2.tar.gz"

data = tarfile.open(filename, "r:gz")
data.extractall()
data.close()

b = open('GRHL1_TCCAAC20NTA_Q_3.fasta','r')
bind = b.readlines()
b.close()

u = open('GRHL1_TCCAAC20NTA_Q_3_shuffled.fasta','r')
unbind = u.readlines()
u.close()

"""### Data preprocessing"""

bind = [v for v in bind if '>' not in v]
bind = [s.replace('\n', '') for s in bind]
bind = [x.upper() for x in bind]

unbind = [v for v in unbind if '>' not in v]
unbind = [s.replace('\n', '') for s in unbind]
unbind = [x.upper() for x in unbind]

print(len(bind), len(unbind))

"""##### Reverse complement"""

bind_rev = list(range(len(bind)))

for i in range(len(bind)):
  seq = Seq(bind[i])
  rev = seq.reverse_complement()
  bind_rev[i] = str(rev)

unbind_rev = list(range(len(unbind)))

for i in range(len(unbind)):
  seq = Seq(unbind[i])
  rev = seq.reverse_complement()
  unbind_rev[i] = str(rev)

bind_fb = bind + bind_rev
unbind_fb = unbind + unbind_rev

bind_label = [1 for i in range(len(bind_fb))]
unbind_label = [0 for i in range(len(unbind_fb))]

bind_dict = {"seq":bind_fb, "label":bind_label}
unbind_dict = {"seq":unbind_fb, "label":unbind_label}

bind_df = pd.DataFrame(bind_dict)
unbind_df = pd.DataFrame(unbind_dict)

df = pd.concat([bind_df, unbind_df])

"""##### split the dataset"""

from sklearn.utils import shuffle

new_df = shuffle(df)
new_df = new_df.reset_index()

x = new_df.seq
y = new_df.label

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=40)

"""##### One-hot Encoding"""

LE = LabelEncoder()
LE.fit(['A', 'C', 'G', 'T', 'N'])

start = datetime.now()

for index, row in x_train.items():
  x_train[index] = LE.transform(list(row))

for index, row in x_test.items():
  x_test[index] = LE.transform(list(row))

x_train = to_categorical(x_train.values.tolist())
x_test = to_categorical(x_test.values.tolist())

y_train = to_categorical(y_train.values.tolist())
y_t = to_categorical(y_test.values.tolist())

end = datetime.now()
print("encoding running time : "+str(end-start))
print("x_train : "+str(len(x_train))+", y_train : "+str(len(y_train)))

"""### GridSearch"""

from keras.metrics import Precision, Recall, TruePositives

def create_model(units=10, neurons=10, activation='relu', filters=10):
    # create model
    model = Sequential()
    model.add(Conv1D(filters, kernel_size=7, strides=1, padding='valid', activation='relu'))
    model.add(MaxPooling1D(pool_size=3, strides=1, padding='valid'))
    model.add(Bidirectional(LSTM(units, input_shape=(20, 5), return_sequences=True)))
    model.add(Bidirectional(LSTM(units)))
    model.add(Dense(neurons, activation=activation))
    model.add(Dropout(0.2))
    model.add(Dense(2, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[TruePositives(name='tp'), Precision(name='precision'), Recall(name='recall'), 'accuracy'])
    return model

# create model
model = KerasClassifier(build_fn=create_model, verbose=0)

# define the grid search parameters

filters = [10, 20, 32, 64]
epochs = [10, 20]
units = [10, 20, 32, 64]
neurons = [10, 20, 32, 64]
activation = ['relu', 'tanh']

param_grid = dict(units=units, epochs=epochs, neurons=neurons, activation=activation, filters=filters)
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring='average_precision', n_jobs=-1, cv=5, verbose=3)

print('------- start grid search -------')

grid_start = datetime.now()
grid_result = grid.fit(x_train, y_train)
grid_end = datetime.now()
print("grid search running time : "+str(grid_end-grid_start))

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
