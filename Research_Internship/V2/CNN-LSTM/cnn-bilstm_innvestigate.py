# -*- coding: utf-8 -*-

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"

import tarfile
import pandas as pd
import numpy as np
import re
from datetime import datetime

import keras
import keras.backend
import keras.layers
import keras.models
import keras.utils
import keras.metrics

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from keras.wrappers.scikit_learn import KerasClassifier

from keras.utils import to_categorical

from matplotlib import pyplot as plt

from Bio.Seq import Seq

import innvestigate
import innvestigate.utils as iutils

keras.__version__

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

df = pd.DataFrame({"seq":bind_fb + unbind_fb, "label":bind_label + unbind_label})

"""##### split the dataset"""

from sklearn.utils import shuffle

new_df = shuffle(df)
new_df = new_df.reset_index()

x = new_df.seq
y = new_df.label

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=40)

"""One-Hot Encoding"""

LE = LabelEncoder()
LE.fit(['A', 'C', 'G', 'T', 'N'])

start = datetime.now()

for index, row in x_train.items():
  x_train[index] = LE.transform(list(row))

for index, row in x_test.items():
  x_test[index] = LE.transform(list(row))

x_train = to_categorical(x_train.values.tolist())
x_t = to_categorical(x_test.values.tolist())

y_train = to_categorical(y_train.values.tolist())
y_t = to_categorical(y_test.values.tolist())

end = datetime.now()
print("encoding running time : "+str(end-start))

"""### LSTM model"""

model = keras.models.Sequential()
model.add(keras.layers.Conv1D(filters=64, kernel_size=7, strides=1, padding='valid', activation='relu'))
model.add(keras.layers.MaxPooling1D(pool_size=3, strides=1, padding='valid'))

model.add(keras.layers.Bidirectional(keras.layers.LSTM(20, return_sequences=True)))
model.add(keras.layers.Bidirectional(keras.layers.LSTM(16, return_sequences=True)))
model.add(keras.layers.Bidirectional(keras.layers.LSTM(10)))
#model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(2, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[keras.metrics.TruePositives(name='tp'), 'accuracy'])

from keras import callbacks
earlystopping = callbacks.EarlyStopping(monitor ="val_tp", mode ="max", patience = 3, restore_best_weights = True)
history = model.fit(x_train, y_train, epochs = 50, validation_split = 0.2, callbacks =[earlystopping])

"""### Visualization

##### select sequences for visualization
TP/FP/TN/FN with/without motif

search sequences with/without motif 'AACCGGTT'
"""

motif_idx = []

for index in range(len(x_t)):
  seqidx_LE = np.argmax(x_t[index], axis=1)
  seqidx = LE.inverse_transform(seqidx_LE)

  for i in range(len(seqidx)-8):
    if (list(seqidx[i:i+8])==['A', 'A', 'C', 'C', 'G', 'G', 'T', 'T']):
      motif_idx.append(index)

wo_motif_idx = [i for i in list(range(len(x_t))) if i not in motif_idx]

"""sequences from positive set with/without motif, 
sequences from negative set with/without motif
"""

motif_idx_neg = []
motif_idx_pos = []

wo_motif_idx_neg = []
wo_motif_idx_pos = []

for i in range(len(motif_idx)):
  if (y_t[motif_idx[i]][0]==1):
    motif_idx_neg.append(motif_idx[i])
  else :
    motif_idx_pos.append(motif_idx[i])

for i in range(len(wo_motif_idx)):
  if (y_t[wo_motif_idx[i]][0]==1):
    wo_motif_idx_neg.append(wo_motif_idx[i])
  else :
    wo_motif_idx_pos.append(wo_motif_idx[i])

"""TP, TN, FP, FN"""

vi_seq = np.zeros(4)

# find seq with motif, TP FP TN FN
for i, idx in enumerate(motif_idx_pos):
  p = model.predict(np.reshape(x_t[idx], (1, 20, 5)))[0,1]
  if p>0.7:
    vi_seq[0] = idx
  if p<0.3:
    vi_seq[1] = idx
  
  if (vi_seq[0] != 0 and vi_seq[1] != 0):
    break

for i, idx in enumerate(motif_idx_neg):
  p = model.predict(np.reshape(x_t[idx], (1, 20, 5)))[0,1]
  if p>0.7:
    vi_seq[2] = idx
  if p<0.3:
    vi_seq[3] = idx
  
  if (vi_seq[2] != 0 and vi_seq[3] != 0):
    vi_seq = vi_seq.astype(int)
    print(vi_seq)
    break

wo_vi_seq = np.zeros(4)

# find seq without motif, TP FP TN FN
for i, idx in enumerate(wo_motif_idx_pos):
  p = model.predict(np.reshape(x_t[idx], (1, 20, 5)))[0,1]
  if p>0.7:
    wo_vi_seq[0] = idx
  if p<0.3:
    wo_vi_seq[1] = idx
  
  if (wo_vi_seq[0] != 0 and wo_vi_seq[1] != 0):
    break

for i, idx in enumerate(wo_motif_idx_neg):
  p = model.predict(np.reshape(x_t[idx], (1, 20, 5)))[0,1]
  if p>0.7:
    wo_vi_seq[2] = idx
  if p<0.3:
    wo_vi_seq[3] = idx
  
  if (wo_vi_seq[2] != 0 and wo_vi_seq[3] != 0):
    wo_vi_seq = wo_vi_seq.astype(int)
    print(wo_vi_seq)
    break

print(vi_seq)
print(wo_vi_seq)

"""##### iNNvestigate"""

def calculate_gradient(method, input_data):
  analyzer = innvestigate.create_analyzer(method[0], model, **method[2])
  title = method[1]
  
  input_seq = np.reshape(input_data, (1, 20, 5))
  analysis = analyzer.analyze(input_seq)
  
  seq_LE = np.argmax(input_data, axis=1)
  seq = LE.inverse_transform(seq_LE)
  # print(seq)
  
  seq_val = np.empty(20)
  grad = analysis.squeeze()
  
  for i in range(20):
    nu = seq_LE[i]
    seq_val[i] = grad[i][nu]

  return seq, seq_val, title

def visualization(seq, seq_val, method, prob, actual, num_for_figtitle):
  title = ['w_TP.png', 'w_FN.png', 'w_FP.png', 'w_TN.png', 'wo_TP.png', 'wo_FN.png', 'wo_FP.png', 'wo_TN.png', 'CNN-BiLSTM1.png', 'CNN-BiLSTM2.png']
  fig, ax = plt.subplots(len(seq), 2, gridspec_kw={'width_ratios': [1, 9]}, figsize=(len(seq), len(seq)-4))

  for i in range(len(seq)):
    ax[i,0].text(0.5, 0.5, method[i], size=10, va="center", color="black")
    ax[i,0].axis('off')
    
    ax[i,1].imshow(np.reshape(seq_val[i], (1,20)), cmap='Blues', interpolation='nearest')
    for j in range(20):
      text = ax[i,1].text(j, 0, seq[i][j], size=12, ha="center", va="center", color="black")
    ax[i,1].axis('off')

  fig.tight_layout()
  plt.axis('off')
  plt.subplots_adjust(hspace = 0.001)
  plt.savefig('cnn_bilstm/'+title[num_for_figtitle])

def save_data(innv_df):
  len_df = len(innv_df)
  
  for i in range(len(methods)):
    innv_df.at[i+len_df, 'seq'] = seq[i]
    innv_df.at[i+len_df, 'method'] = methods[i]
    innv_df.at[i+len_df, 'seq_val'] = seq_val[i]
    innv_df.at[i+len_df, 'prob'] = prob
    innv_df.at[i+len_df, 'actual'] = prob
    
  return innv_df

methods = [
    ("gradient",   "Gradient", {}),
    ("smoothgrad", "SmoothGrad", {}),
    ("deconvnet", "Deconvnet", {}),
    ("guided_backprop", "Guided Backprop", {}),
    ("deep_taylor.bounded", "DeepTaylor", {"low": -1, "high": 1}),
    ("integrated_gradients", "Integrated Gradients", {}),
    ("lrp.z", "LRP-Z", {}),
    ("lrp.epsilon", "LRP-Epsilon", {}),
    ("lrp.sequential_preset_a_flat", "LRP-PresetAFlat", {}),
    ("lrp.sequential_preset_b_flat", "LRP-PresetBFlat", {})
]

innv_df = pd.DataFrame(columns=('seq', 'method', 'seq_val', 'prob', 'actual'))

# for model comparison

seq1 = 'TTGCGGAACCGGTTTTACTA'
seq2 = 'AATCAGGAGTAACCGGTTTC' 

seq1_LE = LE.transform(list(seq1))
seq2_LE = LE.transform(list(seq2))

seq1_input = to_categorical(seq1_LE.tolist())
seq2_input = to_categorical(seq2_LE.tolist())

seq = np.empty([len(methods), 20], dtype="object") 
seq_val = np.empty([len(methods), 20]) 
method = np.empty(len(methods), dtype="object")

for i in range(len(methods)):
  seq[i], seq_val[i], method[i] = calculate_gradient(methods[i], seq1_input)

prob = model.predict(np.reshape(seq1_input, (1, 20, 5)))[0,1]
actual = 1
visualization(seq, seq_val, method, prob, actual, 8)

innv_df = save_data(innv_df)

seq = np.empty([len(methods), 20], dtype="object") 
seq_val = np.empty([len(methods), 20]) 
method = np.empty(len(methods), dtype="object")

for i in range(len(methods)):
  seq[i], seq_val[i], method[i] = calculate_gradient(methods[i], seq2_input)

prob = model.predict(np.reshape(seq2_input, (1, 20, 5)))[0,1]
actual = 1
visualization(seq, seq_val, method, prob, actual, 9)

innv_df = save_data(innv_df)
innv_df.to_pickle('CNN-BiLSTM_innv.pkl')


"""sequence from positive dataset with grhl1 motif"""

# with motif TP
seq = np.empty([len(methods), 20], dtype="object") 
seq_val = np.empty([len(methods), 20]) 
method = np.empty(len(methods), dtype="object")

for i in range(len(methods)):
  seq[i], seq_val[i], method[i] = calculate_gradient(methods[i], x_t[vi_seq[0]])

prob = model.predict(np.reshape(x_t[vi_seq[0]], (1, 20, 5)))[0,1]
actual = 1
visualization(seq, seq_val, method, prob, actual, 0)

# with motif FN
seq = np.empty([len(methods), 20], dtype="object") 
seq_val = np.empty([len(methods), 20]) 
method = np.empty(len(methods), dtype="object")

for i in range(len(methods)):
  seq[i], seq_val[i], method[i] = calculate_gradient(methods[i], x_t[vi_seq[1]])

prob = model.predict(np.reshape(x_t[vi_seq[1]], (1, 20, 5)))[0,1]
actual = 1
visualization(seq, seq_val, method, prob, actual, 1)

"""sequence from negative dataset with grhl1 motif"""

# with motif FP
seq = np.empty([len(methods), 20], dtype="object") 
seq_val = np.empty([len(methods), 20]) 
method = np.empty(len(methods), dtype="object")

for i in range(len(methods)):
  seq[i], seq_val[i], method[i] = calculate_gradient(methods[i], x_t[vi_seq[2]])

prob = model.predict(np.reshape(x_t[vi_seq[2]], (1, 20, 5)))[0,1]
actual = 0
visualization(seq, seq_val, method, prob, actual, 2)

# with motif TN
seq = np.empty([len(methods), 20], dtype="object") 
seq_val = np.empty([len(methods), 20]) 
method = np.empty(len(methods), dtype="object")

for i in range(len(methods)):
  seq[i], seq_val[i], method[i] = calculate_gradient(methods[i], x_t[vi_seq[3]])

prob = model.predict(np.reshape(x_t[vi_seq[3]], (1, 20, 5)))[0,1]
actual = 0
visualization(seq, seq_val, method, prob, actual, 3)

"""sequence from positive dataset without grhl1 motif"""

# without motif TP
seq = np.empty([len(methods), 20], dtype="object") 
seq_val = np.empty([len(methods), 20]) 
method = np.empty(len(methods), dtype="object")

for i in range(len(methods)):
  seq[i], seq_val[i], method[i] = calculate_gradient(methods[i], x_t[wo_vi_seq[0]])

prob = model.predict(np.reshape(x_t[wo_vi_seq[0]], (1, 20, 5)))[0,1]
actual = 1
visualization(seq, seq_val, method, prob, actual, 4)

# without motif FN
seq = np.empty([len(methods), 20], dtype="object") 
seq_val = np.empty([len(methods), 20]) 
method = np.empty(len(methods), dtype="object")

for i in range(len(methods)):
  seq[i], seq_val[i], method[i] = calculate_gradient(methods[i], x_t[wo_vi_seq[1]])

prob = model.predict(np.reshape(x_t[wo_vi_seq[1]], (1, 20, 5)))[0,1]
actual = 1
visualization(seq, seq_val, method, prob, actual, 5)

"""sequence from negative dataset without grhl1 motif"""

# without motif FP
seq = np.empty([len(methods), 20], dtype="object") 
seq_val = np.empty([len(methods), 20]) 
method = np.empty(len(methods), dtype="object")

for i in range(len(methods)):
  seq[i], seq_val[i], method[i] = calculate_gradient(methods[i], x_t[wo_vi_seq[2]])

prob = model.predict(np.reshape(x_t[wo_vi_seq[2]], (1, 20, 5)))[0,1]
actual = 0
visualization(seq, seq_val, method, prob, actual, 6)

# without motif TN
seq = np.empty([len(methods), 20], dtype="object") 
seq_val = np.empty([len(methods), 20]) 
method = np.empty(len(methods), dtype="object")

for i in range(len(methods)):
  seq[i], seq_val[i], method[i] = calculate_gradient(methods[i], x_t[wo_vi_seq[3]])

prob = model.predict(np.reshape(x_t[wo_vi_seq[3]], (1, 20, 5)))[0,1]
actual = 0
visualization(seq, seq_val, method, prob, actual, 7)

"""### Evaluation

Accuracy
"""

score = model.evaluate(x_t, y_t)
print("accuracy = " + str(round(score[2],4)))

"""loss-epoch curve"""

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('CNN-BiLSTM Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'val'], loc='upper right')
plt.savefig("CNN-BiLSTM_loss.png")

"""precision-recall curve"""

probs = model.predict(x_t)[:,1]

precision, recall, thresholds = precision_recall_curve(y_test, probs)

from sklearn.metrics import auc
pr_auc = auc(recall, precision)

"""ROC curve & AUC"""

auc = roc_auc_score(y_test, probs)
fpr, tpr, _ = roc_curve(y_test, probs)

"""save the result"""

result_dict = {'accuracy': score[2], 'loss': history.history['loss'], 'val_loss': history.history['val_loss'],
               'precision': precision, 'recall': recall, 'tpr': tpr, 'fpr': fpr, 'auc': auc, 'pr_auc': pr_auc}
result_df = pd.DataFrame({ key:pd.Series(value) for key, value in result_dict.items() })
result_df.head()

result_df.to_csv ('CNN-BiLSTM_result.csv', index = False, header=True)
