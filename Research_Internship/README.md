# Convolutional Neural Network model for transcription factor binding prediction

### TO DO
- ~~One Hot Encoding takes too long... Maybe other ways to do..? (already tried : sklearn)~~ **The long running time was because of dataframe append function not encoding**
- ~~Input format (pandas series, not numpy array)~~
- ~~How to set appropriate number of nodes, filters and layers?~~ **Hyper parameter tunning, able to do by GridSearch in python (cannot actually run due to running time)**

### Background

#### One Hot Encoding
What is One Hot Encoding?
> *The representation of categorical data as binary vector.* <br/>
> *For example, ['red', 'black', 'green', 'white', 'white'] can be [[0,0,1,0], [1,0,0,0], [0,1,0,0], [0,0,0,1], [0,0,0,1]]*

Why One Hot Encoding is necessary?
> *Most of machine learning methods cannot directly use categorical data as inputs. The label encoding also can be used but machine might recognize it as 'number', then the machine can learn wrong pattern.*

#### 1D CNN
<img src="https://missinglink.ai/wp-content/uploads/2019/03/1D-convolutional-example_2x.png" width=30% height=30%>

> *Conv1D(filter, kernal_size, strides, activation)*
- filter : number of output filters used in operation
- kernal_size : size of window
- strides : shift size of window
- activation : type of activation function (relu, sigmoid, softmax...)

> *MaxPooling1D(pool_size, strides)*
- pool_size : size of window
- strides : shift size of window

> *Flatten()*
- connection between Conv and Dense

> *Dense(units, activation)*
- units : dimensionality of the output
- activation : type of activation function (relu, sigmoid, softmax...)

> *Dropout(rate)*
- rate : fraction of the input units to drop

###### Helpful links
[Keras](https://keras.io/) <br/>
[Tensorflow](https://www.tensorflow.org/api_docs/python/tf/keras/layers) <br/>
[Grid search for deep learning model](https://machinelearningmastery.com/grid-search-hyperparameters-deep-learning-models-python-keras/)
