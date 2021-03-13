# Recurrent neural network

### TO DO
- Grid search
- Embedding?

### Background

##### Word Embedding
- the method represents the 'word' as a vector <br/>
- can represent semantic relationship between the words

##### RNN
<img src="https://cdn.analyticsvidhya.com/wp-content/uploads/2020/02/assets_-LvBP1svpACTB1R1x_U4_-LwEQnQw8wHRB6_2zYtG_-LwEZT8zd07mLDuaQZwy_image-1.png" width=50% height=50%> <br/>
- network contains at least one feed-back connection <br/>
- Unlikely feedforward NN, RNN send back the output from previous step as an input of network. <br/>
- can capture sequential information, such as context of sentences. <br/>
- suitable for text, audio data and time-series analysis <br/>

##### LSTM
<img src="http://i.imgur.com/jKodJ1u.png" width=50% height=50%> <br/>

- Long Short Term Memory <br/>
- more useful to remember previous data <br/>
- modified version of RNN (without vanishing gradient problem) <br/>


`what is vanishing gradient problem?`
As learning continues, the effect of the information received earlier gradually decreases and eventually can disappear

##### Bidirectional LSTM
<img src="https://i.imgur.com/fLc4u4w.png" width=50% height=50%> <br/>

- put two independent RNNs together; one reads forward and another reads backward <br/>

> **Embedding layer**
> learns embedding of each word via input <br/>
> Inputs have to be represented by unique integer (conversion is done by keras.Tokenizer) <br/>
> Words will be 7-mers in our data <br/>

`Embedding(input_dim, output_dim, input_length)`
- input_dim : number of words in data
- output_dim : size of vector words will be embedded
- input_length : number of words in one input (14 in our data since we have 20bp length sequences)

> **LSTM layer**

`LSTM(units, actiavtion, return_sequences)`
- units : dimentionality of output
- activation : activation function
- return_sequences : default is 'False'. Set 'True' if the layer is followed by another LSTM layer.
