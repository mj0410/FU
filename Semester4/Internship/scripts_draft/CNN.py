import tarfile
import pandas as pd
import numpy as np
import re

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv1D, MaxPooling1D

def onehot_encoder(seq):
	nu = ['[aA]','[cC]','[gG]','[tT]','[nN]']
	for i in range(len(nu)):
		index = [m.start() for m in re.finditer(nu[i], seq)]
		loc = np.zeros(len(seq),dtype=np.int)
		loc[index] = 1
		if i==0:
			mat = np.zeros((len(seq),5))
		mat[...,i] = loc
	return mat


##### Read File #####
    
filename = "minie_training_data.tar.gz"

data = tarfile.open(filename, "r:gz")
data.extractall()
data.close()

b = open('ghl_gold.fa','r')
bind = b.readlines()
b.close()

u = open('ghl_gold_random.fa','r')
unbind = u.readlines()
u.close()

bind = [v for v in bind if '>' not in v]
bind = [s.replace('\n', '') for s in bind]
bind = bind[:100000]

unbind = [v for v in unbind if '>' not in v]
unbind = [s.replace('\n', '') for s in unbind]
unbind = unbind[:100000]


##### OneHot Encoding #####

df = pd.DataFrame({'seq':[], 'label':[]})

for i in range(len(bind)):
  mat = onehot_encoder(bind[i])
  df = df.append({'seq':mat.tolist(), 'label':1}, ignore_index=True)

for i in range(len(unbind)):
  mat = onehot_encoder(unbind[i])
  df = df.append({'seq':mat.tolist(), 'label':0}, ignore_index=True)

##### Data Preprocessing #####

new_df = shuffle(df)
new_df = new_df.reset_index(drop=True)

x = new_df.seq
y = new_df.label

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=40)

x_train = np.asarray(x_train.to_list())
x_test = np.asarray(x_test.to_list())

data1 = np.array(y_train.values.tolist())
y_train = to_categorical(data1)
data2 = np.array(y_test.values.tolist())
y_test = to_categorical(data2)

x_train = x_train.astype('float32')


##### CNN model #####

model=Sequential()
model.add(Conv1D(filters=64, kernel_size=3, strides=1, padding='valid', input_shape=(20,5), activation='relu'))
model.add(MaxPooling1D(pool_size=3, strides=1, padding='valid'))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(2, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5, validation_data= None)


##### Evaluation #####

score = model.evaluate(x_test, y_test, verbose=1)
print("score = " + str(score))

