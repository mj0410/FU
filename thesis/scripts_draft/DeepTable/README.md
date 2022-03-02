## DeepTable (table orientation classification)

```
├── run.sh
├── DeepTableEval.py
├── DeepTablePred.py
├── DeepTableTrain.py
├── input_transformation.py
├── preprocessing.py
├── results                   <- evaluation / prediction results
└── tables.pickle             <- tables for training the model
```

##### run.sh
```
export BASEPATH=PATH_TO_DEEPTABLE
export INPUT_DIR=PATH_TO_INPUT_TABLES
export EMBEDDING=$BASEPATH/EMBEDDING_FILE
export MODEL_DIR=$BASEPATH/models
export TRAINING_TABLES=$BASEPATH/tables.pickle

python DeepTableTrain.py -e 50 -l 0.01 -v $EMBEDDING -i $TRAINING_TABLES -o $MODEL_DIR
python DeepTableEval.py -m $MODEL_DIR/TRAINED_MODEL_FILE -i $INPUT_DIR -o OUTPUT_NAME
python DeepTablePred.py -m $MODEL_DIR/TRAINED_MODEL_FILE -i $INPUT_DIR -o OUTPUT_NAME
```

### To do

- [ ] Write a code to transform table based on predicted label
