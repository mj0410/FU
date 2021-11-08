#!/usr/bin/env python
# coding: utf-8

# In[5]:


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
from keras.layers import Dropout, Activation, Flatten
from keras.layers import Embedding, LSTM, Dense, Bidirectional

from matplotlib import pyplot as plt

from Bio.Seq import Seq


# ### Read data

# In[6]:


filename = "training_data.tar.gz"

data = tarfile.open(filename, "r:gz")
data.extractall()
data.close()


# In[7]:


b = open('ghl_gold.fa','r')
bind = b.readlines()
b.close()

u = open('ghl_gold_random.fa','r')
unbind = u.readlines()
u.close()


# ### Data preprocessing

# In[8]:


bind = [v for v in bind if '>' not in v]
bind = [s.replace('\n', '') for s in bind]
bind = [x.upper() for x in bind]

unbind = [v for v in unbind if '>' not in v]
unbind = [s.replace('\n', '') for s in unbind]
unbind = [x.upper() for x in unbind]


# In[9]:


print(len(bind), len(unbind))


# Reverse Complement

# In[10]:


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


# In[11]:


bind_fb = bind + bind_rev
unbind_fb = unbind + unbind_rev


# In[12]:


bind_label = [1 for i in range(len(bind_fb))]
unbind_label = [0 for i in range(len(unbind_fb))]


# In[13]:


df = pd.DataFrame({"seq": bind_fb + unbind_fb, "label":bind_label + unbind_label})


# ##### split the dataset

# In[14]:


from sklearn.utils import shuffle

new_df = shuffle(df)
new_df = new_df.reset_index()


# In[15]:


x = new_df.seq
y = new_df.label


# In[16]:


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=40)


# One-Hot Encoding

# In[17]:


LE = LabelEncoder()
LE.fit(['A', 'C', 'G', 'T', 'N'])


# In[18]:


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


# ### RNN model

# LSTM

# In[19]:


model = keras.models.Sequential()
model.add(keras.layers.LSTM(128, input_shape=(20, 5), return_sequences=True))
model.add(keras.layers.LSTM(128))
model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Dense(2, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# In[20]:


model.summary()


# In[21]:


history = model.fit(x_train, y_train, epochs = 10, validation_split = 0.2)


# BiLSTM

# In[ ]:


bi_model = keras.layers.Sequential()
bi_model.add(keras.layers.Bidirectional(keras.layers.LSTM(64, return_sequences=True), input_shape=(20, 5)))
bi_model.add(keras.layers.Bidirectional(keras.layers.LSTM(64)))
bi_model.add(keras.layers.Dense(64, activation='relu'))
bi_model.add(keras.layers.Dropout(0.2))
bi_model.add(keras.layers.Dense(2, activation='sigmoid'))
bi_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# In[ ]:


bi_model.summary()


# In[ ]:


bi_history = bi_model.fit(x_train, y_train, epochs = 10, validation_split = 0.2)


# ### Evaluation

# Accuracy

# In[ ]:


score = model.evaluate(x_t, y_t)
print("score = " + str(round(score[1],2)))


# In[ ]:


bi_score = bi_model.evaluate(x_test, y_test)
print("score = " + str(round(bi_score[1],2)))


# loss-epoch curve

# In[ ]:


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'val'], loc='upper right')
plt.show()


# In[ ]:


plt.plot(bi_history.history['loss'])
plt.plot(bi_history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'val'], loc='upper right')
plt.show()


# precision-recall curve

# In[ ]:


probs = model.predict(x_t)[:,1]
bi_probs = bi_model.predict(x_t)[:,1]


# In[ ]:


precision, recall, thresholds = precision_recall_curve(y_test, probs)
bi_precision, bi_recall, bi_thresholds = precision_recall_curve(y_test, bi_probs)


# In[ ]:


plt.plot(recall, precision)

plt.title('LSTM Precision-Recall Curve')

plt.xlabel('Recall')
plt.ylabel('Precision')

plt.show()


# In[ ]:


plt.plot(bi_recall, bi_precision)

plt.title('BiLSTM Precision-Recall Curve')

plt.xlabel('Recall')
plt.ylabel('Precision')

plt.show()


# ROC curve & AUC

# In[ ]:


auc = roc_auc_score(y_test, probs)
fpr, tpr, _ = roc_curve(y_test, probs)

bi_auc = roc_auc_score(y_test, bi_probs)
bi_fpr, bi_tpr, bi_ = roc_curve(y_test, bi_probs)


# In[ ]:


plt.plot(fpr, tpr)
plt.title('LSTM ROC Curve (AUC = ' + str(round(auc,2)) + ')')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()


# In[ ]:


plt.plot(bi_fpr, bi_tpr)
plt.title('BiLSTM ROC Curve (AUC = ' + str(round(bi_auc,2)) + ')')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()


# save the result

# In[ ]:


result_dict = {'accuracy': score[1], 'loss': history.history['loss'], 'val_loss': history.history['val_loss'],
               'precision': precision, 'recall': recall, 'tpr': tpr, 'fpr': fpr, 'auc': auc}
result_df = pd.DataFrame({ key:pd.Series(value) for key, value in result_dict.items() })
result_df.head()


# In[ ]:


result_df.to_csv ('LSTM_result.csv', index = False, header=True)


# In[ ]:


bi_result_dict = {'accuracy': bi_score[1], 'loss': bi_history.history['loss'], 'val_loss': bi_history.history['val_loss'],
               'precision': bi_precision, 'recall': bi_recall, 'tpr': bi_tpr, 'fpr': bi_fpr, 'auc': bi_auc}
bi_result_df = pd.DataFrame({ key:pd.Series(value) for key, value in bi_result_dict.items() })
bi_result_df.head()


# In[ ]:


bi_result_df.to_csv ('BiLSTM_result.csv', index = False, header=True)

