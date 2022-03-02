export BASEPATH=C:/Users/minie/Desktop/git/MS_THESIS/scripts_draft/Sato
export INPUT_DIR=C:/Users/minie/Desktop/git/MS_THESIS/DBs/Synthea
export SHERLOCKPATH=$BASEPATH/sherlock
export EXTRACTPATH=$BASEPATH/extract
export PYTHONPATH=$PYTHONPATH:$SHERLOCKPATH
export PYTHONPATH=$PYTHONPATH:$BASEPATH
export TYPENAME='type_synthea'
export TOPICMODELPATH=$BASEPATH/topic_model/LDA_cache/$TYPENAME
export LDA_name='row50_16'

#cd $BASEPATH/extract
#python make_type_list.py --file_path $INPUT_DIR --type_name type_synthea #ok
#python extract_header.py valid_header_row50.pkl -n 50 -o True #ok
#python split_train_test.py --header_file valid_header_row50.pkl --output split_row50 --cv 5 # --val_percent 20 --test_percent 20 #ok
#python extract_features.py valid_header_row50.pkl -O row50_100 -f topic -LDA row50_100  #ok

cd $BASEPATH/model
#python train_sherlock.py -c $BASEPATH/configs/train_sherlock_config.txt
#python train_sherlock.py -c $BASEPATH/configs/eval_sherlock_config.txt
#python pred_sherlock.py -c $BASEPATH/configs/pred_sherlock_config.txt
# python sherlock_exp.py -c $BASEPATH/configs/cv_sherlock_topic_config.txt

python train_CRF_LC.py -c $BASEPATH/configs/train_topic_config.txt