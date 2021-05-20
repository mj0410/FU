### Neural Network

- inspired by the neural architecture of a human brain
- The output is provided by 'neurons' by applying functions (activation function) on the inputs.

````diff
Activation functions
# sigmoid
- captures non-linearity in the data
- problem of vanishing gradients. 
# Tanh
- rescaled version of the sigmoid
- problem of vanishing gradients
# ReLU
- The Rectified Linear Unit
- The most commonly used activation function
- f(x)=max (0, x)
````
##### Overfitting in ML
- when the model learns the training data too well
- **How to avoid?** make model less complex, add dropout layer, early stopping

### ANN / CNN / RNN 
##### ANN (Artificial Neural Network) 
- input layer / hidden layers / output layer 
- Feed-Forward Neural network 
- each layer learns certain weights
##### CNN (Convolutional Neural Network)
- convolution / pooling / fully-connected layers
- captures spatial features
- image / video data
##### RNN (Recurrent Neural Network)
- recurrent connection on the hidden layers
- captures context
- time-series / text / audio data

### Transcription Factor
- Transcription factors bind to specific DNA sequences, open promoter and enhancer regions, which play important role in regulating gene expression.
- PWM is one of the most commonly used method to predict transcription factor binding site.

### position weight matrix (PWM)
<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/8/85/LexA_gram_positive_bacteria_sequence_logo.png/440px-LexA_gram_positive_bacteria_sequence_logo.png" width=50% height=50%> <br/>
- representation of motifs in biological sequences
- Each letter (nucleotide) has score, which represents how well the sequence matches a motif (transcription factor binding site in our case).
```diff
# Advantage
+ simple model, easy to construct
+ can intuitively express the binding affinity with fewer parameters

# Disadvantage
- sensitive to quality and size of set of TFBS sequences
- high false positive rate (The model assumes independence between base positions of sequence)
- do not include composition or structure of TFs
```
### Grainyhead-like 1
<img src="http://jaspar.genereg.net/static/logos/svg/MA0647.1.svg" width=50% height=50%> <br/>
