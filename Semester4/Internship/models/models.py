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
from keras.wrappers.scikit_learn import KerasClassifier

from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dropout, Activation, Flatten, Dense
from keras.layers import Embedding, LSTM, Dense, Bidirectional, Conv1D, MaxPooling1D
from keras.metrics import TruePositives

from matplotlib import pyplot as plt

from Bio.Seq import Seq
from Bio import SeqIO

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

"""Reverse Complement"""

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

"""### CNN model"""

from keras import callbacks
earlystopping = callbacks.EarlyStopping(monitor ="val_loss", mode ="min", patience = 3, restore_best_weights = True)

print('CNN model training')

cnn = Sequential()
cnn.add(Conv1D(filters=128, kernel_size=7, strides=1, padding='valid', activation='relu'))
cnn.add(MaxPooling1D(pool_size=3, strides=1, padding='valid'))
cnn.add(Flatten())
cnn.add(Dense(64, activation='relu'))
cnn.add(Dropout(0.5))
cnn.add(Dense(2, activation='sigmoid'))
cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=[TruePositives(name='tp'), 'accuracy'])

cnn_start = datetime.now()

cnn_history = cnn.fit(x_train, y_train, epochs = 50, validation_split = 0.2, callbacks =[earlystopping], verbose=0)

cnn_end = datetime.now()
print("CNN training running time : "+str(cnn_end-cnn_start))

cnn_score = cnn.evaluate(x_test, y_t)
cnn_probs = cnn.predict(x_test)[:,1]
cnn_precision, cnn_recall, cnn_thresholds = precision_recall_curve(y_test.values, cnn_probs)

from sklearn.metrics import auc
cnn_pr_auc = auc(cnn_recall, cnn_precision)

cnn_auc = roc_auc_score(y_test.values, cnn_probs)
cnn_fpr, cnn_tpr, cnn_ = roc_curve(y_test.values, cnn_probs)

cnn_result_dict = {'accuracy': cnn_score[2], 'loss': cnn_history.history['loss'], 'val_loss': cnn_history.history['val_loss'],
               'precision': cnn_precision, 'recall': cnn_recall, 'tpr': cnn_tpr, 'fpr': cnn_fpr, 'auc': cnn_auc, 'pr_auc': cnn_pr_auc}
cnn_result_df = pd.DataFrame({ key:pd.Series(value) for key, value in cnn_result_dict.items() })

cnn_result_df.to_csv ('CNN_result.csv', index = False, header=True)

"""### RNN model

LSTM
"""

lstm = Sequential()
lstm.add(LSTM(16, return_sequences=True))
#lstm.add(LSTM(64, return_sequences=True))
lstm.add(LSTM(16))
lstm.add(Dense(10, activation='relu'))
lstm.add(Dropout(0.2))
lstm.add(Dense(2, activation='sigmoid'))
lstm.compile(optimizer='adam', loss="binary_crossentropy", metrics=[TruePositives(name='tp'), 'accuracy'])

lstm_start = datetime.now()

lstm_history = lstm.fit(x_train, y_train, epochs = 50, validation_split = 0.2, callbacks =[earlystopping], verbose=0)

lstm_end = datetime.now()
print("LSTM training running time : "+str(lstm_end-lstm_start))

lstm_score = lstm.evaluate(x_test, y_t)
lstm_probs = lstm.predict(x_test)[:,1]
lstm_precision, lstm_recall, lstm_thresholds = precision_recall_curve(y_test.values, lstm_probs)

lstm_pr_auc = auc(lstm_recall, lstm_precision)

lstm_auc = roc_auc_score(y_test.values, lstm_probs)
lstm_fpr, lstm_tpr, lstm_ = roc_curve(y_test.values, lstm_probs)

lstm_result_dict = {'accuracy': lstm_score[2], 'loss': lstm_history.history['loss'], 'val_loss': lstm_history.history['val_loss'],
               'precision': lstm_precision, 'recall': lstm_recall, 'tpr': lstm_tpr, 'fpr': lstm_fpr, 'auc': lstm_auc, 'pr_auc': lstm_pr_auc}
lstm_result_df = pd.DataFrame({ key:pd.Series(value) for key, value in lstm_result_dict.items() })

lstm_result_df.to_csv ('LSTM_result.csv', index = False, header=True)

"""BiLSTM"""

bilstm = Sequential()
bilstm.add(Bidirectional(LSTM(16, return_sequences=True)))
#bilstm.add(Bidirectional(LSTM(64, return_sequences=True)))
bilstm.add(Bidirectional(LSTM(16)))
bilstm.add(Dense(32, activation='relu'))
bilstm.add(Dropout(0.2))
bilstm.add(Dense(2, activation='sigmoid'))
bilstm.compile(optimizer='adam', loss="binary_crossentropy", metrics=[TruePositives(name='tp'), 'accuracy'])

bilstm_start = datetime.now()

bilstm_history = bilstm.fit(x_train, y_train, epochs = 50, validation_split = 0.2, callbacks =[earlystopping], verbose=0)

bilstm_end = datetime.now()
print("BiLSTM training running time : "+str(bilstm_end-bilstm_start))

bilstm_score = bilstm.evaluate(x_test, y_t)
bilstm_probs = bilstm.predict(x_test)[:,1]
bilstm_precision, bilstm_recall, bilstm_thresholds = precision_recall_curve(y_test.values, bilstm_probs)

bilstm_pr_auc = auc(bilstm_recall, bilstm_precision)

bilstm_auc = roc_auc_score(y_test.values, bilstm_probs)
bilstm_fpr, bilstm_tpr, bilstm_ = roc_curve(y_test.values, bilstm_probs)

bilstm_result_dict = {'accuracy': bilstm_score[2], 'loss': bilstm_history.history['loss'], 'val_loss': bilstm_history.history['val_loss'],
               'precision': bilstm_precision, 'recall': bilstm_recall, 'tpr': bilstm_tpr, 'fpr': bilstm_fpr, 'auc': bilstm_auc, 'pr_auc': bilstm_pr_auc}
bilstm_result_df = pd.DataFrame({ key:pd.Series(value) for key, value in bilstm_result_dict.items() })

bilstm_result_df.to_csv ('BiLSTM_result.csv', index = False, header=True)

"""### CNN-RNN model

CNN-LSTM
"""

cl = Sequential()
cl.add(Conv1D(filters=64, kernel_size=7, strides=1, padding='valid', activation='relu'))
cl.add(MaxPooling1D(pool_size=3, strides=1, padding='valid'))

cl.add(LSTM(16, return_sequences=True))
cl.add(LSTM(16))
cl.add(Dense(10, activation='relu'))
cl.add(Dropout(0.5))
cl.add(Dense(2, activation='sigmoid'))

cl.compile(optimizer='adam', loss="binary_crossentropy", metrics=[TruePositives(name='tp'), 'accuracy'])

cl_start = datetime.now()

cl_history = cl.fit(x_train, y_train, epochs = 50, validation_split = 0.2, callbacks =[earlystopping], verbose=0)

cl_end = datetime.now()
print("CNN-LSTM training running time : "+str(cl_end-cl_start))

cl_score = cl.evaluate(x_test, y_t)
cl_probs = cl.predict(x_test)[:,1]
cl_precision, cl_recall, cl_thresholds = precision_recall_curve(y_test.values, cl_probs)

cl_pr_auc = auc(cl_recall, cl_precision)

cl_auc = roc_auc_score(y_test.values, cl_probs)
cl_fpr, cl_tpr, cl_ = roc_curve(y_test.values, cl_probs)

cl_result_dict = {'accuracy': cl_score[2], 'loss': cl_history.history['loss'], 'val_loss': cl_history.history['val_loss'],
               'precision': cl_precision, 'recall': cl_recall, 'tpr': cl_tpr, 'fpr': cl_fpr, 'auc': cl_auc, 'pr_auc': cl_pr_auc}
cl_result_df = pd.DataFrame({ key:pd.Series(value) for key, value in cl_result_dict.items() })

cl_result_df.to_csv ('CNN-LSTM_result.csv', index = False, header=True)

"""CNN-BiLSTM"""

cbi = Sequential()
cbi.add(Conv1D(filters=64, kernel_size=7, strides=1, padding='valid', activation='relu'))
cbi.add(MaxPooling1D(pool_size=3, strides=1, padding='valid'))

cbi.add(Bidirectional(LSTM(16, return_sequences=True)))
cbi.add(Bidirectional(LSTM(16)))
cbi.add(Dense(16, activation='relu'))
cbi.add(Dropout(0.5))
cbi.add(Dense(2, activation='sigmoid'))

cbi.compile(optimizer='adam', loss="binary_crossentropy", metrics=[TruePositives(name='tp'), 'accuracy'])

cbi_start = datetime.now()
cbi_history = cbi.fit(x_train, y_train, epochs = 50, validation_split = 0.2, callbacks =[earlystopping], verbose=0)
cbi_end = datetime.now()
print("CNN-BiLSTM training running time : "+str(cbi_end-cbi_start))

cbi_score = cbi.evaluate(x_test, y_t)
cbi_probs = cbi.predict(x_test)[:,1]
cbi_precision, cbi_recall, cbi_thresholds = precision_recall_curve(y_test.values, cbi_probs)

cbi_pr_auc = auc(cbi_recall, cbi_precision)

cbi_auc = roc_auc_score(y_test.values, cbi_probs)
cbi_fpr, cbi_tpr, cbi_ = roc_curve(y_test.values, cbi_probs)

cbi_result_dict = {'accuracy': cbi_score[2], 'loss': cbi_history.history['loss'], 'val_loss': cbi_history.history['val_loss'],
               'precision': cbi_precision, 'recall': cbi_recall, 'tpr': cbi_tpr, 'fpr': cbi_fpr, 'auc': cbi_auc, 'pr_auc': cbi_pr_auc}
cbi_result_df = pd.DataFrame({ key:pd.Series(value) for key, value in cbi_result_dict.items() })

cbi_result_df.to_csv ('CNN-BiLSTM_result.csv', index = False, header=True)
