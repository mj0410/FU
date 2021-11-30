# -*- coding: utf-8 -*-

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"

import tarfile
import pandas as pd
import numpy as np
import re
from datetime import datetime

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import auc

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dropout, Activation, Flatten
from keras.layers import LSTM, Dense
from keras.metrics import TruePositives

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

"""### GridSearch"""

def create_model(units=32, neurons=32, activation='relu'):
    # create model
    model = Sequential()
    model.add(LSTM(units, return_sequences=True))
    model.add(LSTM(units, return_sequences=True))
    model.add(LSTM(units))
    model.add(Dense(neurons, activation=activation))
    model.add(Dropout(0.2))
    model.add(Dense(2, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[TruePositives(name='tp'), 'accuracy'])
    return model

# create model
model = KerasClassifier(build_fn=create_model, verbose=0)

# define the grid search parameters
units = [64, 128, 256]
neurons = [20, 32, 64]
activation = ['relu', 'tanh']

param_grid = dict(neurons=neurons, units=units, activation=activation)
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring='average_precision', n_jobs=-1, cv=5, verbose=3)

grid_start = datetime.now()
grid_result = grid.fit(x_train, y_train)
grid_end = datetime.now()
print("grid search running time : "+str(grid_end-grid_start))

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

"""### Model training"""

params = grid_result.best_params_
lstm = create_model(units=params['units'], neurons=params['neurons'], activation=params['activation'])
history = lstm.fit(x_train, y_train, epochs = 10, validation_split = 0.2, verbose=2)

score = lstm.evaluate(x_test, y_t, verbose=0)

"""precision-recall curve"""

probs = lstm.predict(x_test)[:,1]
precision, recall, thresholds = precision_recall_curve(y_test.values, probs)
pr_auc = auc(recall, precision)

"""ROC curve & AUC"""

auc = roc_auc_score(y_test.values, probs)
fpr, tpr, _ = roc_curve(y_test.values, probs)

"""save the result"""

result_dict = {'accuracy': score[2], 'loss': history.history['loss'], 'val_loss': history.history['val_loss'],
               'precision': precision, 'recall': recall, 'tpr': tpr, 'fpr': fpr, 'auc': auc, 'pr_auc': pr_auc}
result_df = pd.DataFrame({ key:pd.Series(value) for key, value in result_dict.items() })

result_df.to_csv ('LSTM_result.csv', index = False, header=True)
