export BASEPATH=C:/Users/minie/Desktop/git/MS_THESIS/scripts_draft/Sato
export INPUT_DIR=C:/Users/minie/Desktop/git/MS_THESIS/DBs/Synthea/1K
export SHERLOCKPATH=$BASEPATH/sherlock
export EXTRACTPATH=$BASEPATH/extract
export PYTHONPATH=$PYTHONPATH:$SHERLOCKPATH
export PYTHONPATH=$PYTHONPATH:$BASEPATH
export TYPENAME='reduced_synthea'
export LDANAME='row50_16_cv'
export LDA_name='row50_16_cv'
export TOPICMODELPATH=$BASEPATH/topic_model/LDA_cache/$TYPENAME

#cd $BASEPATH/topic_model
#python train_LDA.py -n row50_16_cv -cv 1000 -s split_row50 -b 512 --topic_num 16
#python train_LDA.py -n row50_77 -s split_row50 -b 512 --topic_num 77

cd $BASEPATH/extract_woNone
#python make_type_list.py --file_path $INPUT_DIR --type_name 80nan_synthea #ok
python extract_header.py valid_header_row50_woNan.pkl -n 50 -o True #ok
python split_train_test.py --header_file valid_header_row50_woNan.pkl --output split_row50_woNan --cv 1000

#python extract_features.py valid_header_row50.pkl -O row50_16_cv -f topic -LDA row50_16_cv
#python extract_features.py valid_header_row50.pkl -O row50_77 -f topic -LDA row50_77

#cd $BASEPATH/model
#python train_sherlock.py -c $BASEPATH/configs/train_sherlock_config.txt
#python train_sherlock.py -c $BASEPATH/configs/eval_sherlock_config.txt
#python train_CRF_LC.py -c $BASEPATH/configs/train_topic_config.txt
#python train_CRF_LC.py -c $BASEPATH/configs/eval_topic_config.txt
#python cv_CRF_LC.py -c $BASEPATH/configs/cv_topic_config.txt