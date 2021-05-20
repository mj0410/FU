### Data v2
```diff
+ HT-SELEX GRHL1 Q3
- dinucleotide-preserving shuffle
```
[DeepBind](https://www.nature.com/articles/nbt.3300)

### Model Comparison
##### Loss-Epoch Curve
<img src="https://user-images.githubusercontent.com/66175878/118808154-73fee300-b8a9-11eb-88ab-a9c7acf328c8.png"> <br/>

##### Precision-Recall Curve
<img src="https://user-images.githubusercontent.com/66175878/118808258-9ee93700-b8a9-11eb-8b42-aa6e9f6ffd9c.png" width=30% height=30%> <br/>

##### ROC Curve
<img src="https://user-images.githubusercontent.com/66175878/118808378-c04a2300-b8a9-11eb-832c-85dd3ef1a72a.png" width=30% height=30%> <br/>

### Visualization


### Hyperparameter tuning
##### By GridSearchCV
- 5-fold cross validation
- scoring by roc auc

```
GridSearchCV shows the best model in terms of 'score' not considering overfitting
-> the more complex model, the better AUC (but overfitting can happen when the model is too complex)
```

##### By manual comparison
- look at loss-epoch curve to avoid overfitting
- scoring by roc auc

