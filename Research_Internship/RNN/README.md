# Recurrent neural network

### TO DO
- Grid search
- Embedding?

### Background

##### Word Embedding
- the method represents the 'word' as a vector
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
