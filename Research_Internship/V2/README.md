### Data v2
```diff
+ HT-SELEX GRHL1 Q3
- dinucleotide-preserving shuffle
```
[DeepBind](https://www.nature.com/articles/nbt.3300)

### Model Comparison
##### Loss-Epoch Curve
<img src="https://user-images.githubusercontent.com/66175878/119225836-8b86d780-bb06-11eb-93b5-1ebb0ff98bcb.png"> <br/>

##### Precision-Recall Curve
<img src="https://user-images.githubusercontent.com/66175878/119225869-a8230f80-bb06-11eb-8586-20ffeb86708e.png" width=30% height=30%> <br/>

##### ROC Curve
<img src="https://user-images.githubusercontent.com/66175878/119225897-bf61fd00-bb06-11eb-9005-0081fc1eb2f6.png" width=30% height=30%> <br/>

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

