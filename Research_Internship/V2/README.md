### Data v2
```
+ HT-SELEX GRHL1 Q3
- dinucleotide-preserving shuffle
```
[DeepBind](https://www.nature.com/articles/nbt.3300)

### Model Comparison
##### Loss-Epoch Curve

##### Precision-Recall Curve

##### ROC Curve


### GridSearchCV
- 5-fold cross validation
- scoring by average precision

##### CNN Model

> GridSearch result
```
Best: 0.748417 using {'activation': 'relu', 'filters': 256, 'neurons': 256}
```

##### LSTM Model
> LSTM GridSearch result
```
Best: 0.744748 using {'activation': 'tanh', 'neurons': 32, 'units': 128}
```

> Bidirectional LSTM GridSearch result
```
Best: 0.755622 using {'activation': 'relu', 'neurons': 128, 'units': 200}
```

##### CNN-LSTM Model

> CNN-LSTM GridSearch result
```
Best: 0.743196 using {'activation': 'tanh', 'filters': 64, 'neurons': 64, 'units': 64}
```

> CNN-BiLSTM GridSearch result
```
Best: 0.744849 using {'activation': 'relu', 'filters': 64, 'neurons': 64, 'units': 64}
```


