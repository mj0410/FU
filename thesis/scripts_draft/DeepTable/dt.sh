export SATOPATH=C:/Users/minie/Desktop/git/MS_THESIS/scripts_draft/Sato
export BASEPATH=C:/Users/minie/Desktop/git/MS_THESIS/scripts_draft/DeepTable
export EMBEDDING=C:/Users/minie/Desktop/FU/Thesis/w2v_embedding.bin
export MODEL_DIR=$BASEPATH/models/train_test
export INPUT_TABLES=$BASEPATH/tables.pickle
export TRAIN_TABLES=$BASEPATH/new_tables.pickle
export INPUT=C:/Users/minie/Desktop/git/MS_THESIS/data
#export SPLIT=C:/Users/minie/Desktop/git/MS_THESIS/scripts_draft/Sato/extract/out/train_test_split/DT_train_5fold_synthea.csv
export PRED_DIR=$BASEPATH/prediction
export SPLIT=C:/Users/minie/Desktop/git/MS_THESIS/scripts_draft/Sato/extract/out/train_test_split/test_split_5fold_reduced_synthea.csv

#python tables_for_train.py

#python DeepTableTrain.py -e 100 -l 0.01 -v $EMBEDDING -i $TRAIN_TABLES -o $MODEL_DIR
#python DeepTableEval.py -m $MODEL_DIR/model_99.hdf5 -if $BASEPATH/test.pkl -o evaluation

python DeepTablePred.py -m $MODEL_DIR/model_99.hdf5 -i $INPUT -o synthea100p
#python DeepTablePred.py -m $MODEL_DIR/model_100.hdf5 -s split_row50_1000fold_synthea -o synthea_DTpred_test_new_random
# about 80% / random 64%
#python DeepTablePred.py -m $MODEL_DIR/model_100.hdf5 -s split_row50_1000fold_reduced_synthea -o reduced_synthea_DTpred_test_new
# about 83%
#python DeepTablePred.py -m $MODEL_DIR/model_100.hdf5 -s row100_1000fold_reduced_synthea -o row100_random
# about 85% / random 72%
#python DeepTablePred.py -m $MODEL_DIR/model_100.hdf5 -s row200_1000fold_reduced_synthea -o row200_random
# about 81%
