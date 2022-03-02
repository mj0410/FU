export BASEPATH=C:/Users/minie/Desktop/git/MS_THESIS/scripts_draft/Sato
export INPUT_DIR=C:/Users/minie/Desktop/git/MS_THESIS/DBs/Synthea/1K
export SHERLOCKPATH=$BASEPATH/sherlock
export EXTRACTPATH=$BASEPATH/extract
export PYTHONPATH=$PYTHONPATH:$SHERLOCKPATH
export PYTHONPATH=$PYTHONPATH:$BASEPATH
export TYPENAME='1K_new'
export TOPICMODELPATH=$BASEPATH/topic_model/LDA_cache/$TYPENAME

cd $BASEPATH/extract
#python make_type_list.py --file_path $INPUT_DIR --type_name 1K_oneid #ok
#python extract_header2.py header_row50_1K_rcsv.pkl -n 50 -o True
#python split_train_test.py --header_file header_row50_1K_rcsv.pkl --output row50_1K_rcsv --cv 1000

#python extract_features2.py header_row50_1K_rcsv.pkl -O row50_1K_rcsv

#python extract_features.py header_row50_1K_rcsvr.pkl -O row50_1K_rcsvr

export LDANAME='row50_16_reduced'
export LDA_name='row50_16_reduced'
#cd $BASEPATH/topic_model
#python train_LDA.py -n row50_16_reduced -cv 1000 -s row50_1K_rcsvr -b 512 --topic_num 16
#cd $BASEPATH/extract
#python extract_features.py header_row50_1K_rcsvr.pkl -O row50_16_reduced -f topic -LDA row50_16_reduced

#export LDANAME='row50_8_reduced'
#export LDA_name='row50_8_reduced'
#cd $BASEPATH/topic_model
#python train_LDA.py -n row50_8_reduced -cv 1000 -s row50_1K_rcsvr -b 512 --topic_num 8
#cd $BASEPATH/extract
#python extract_features.py header_row50_1K_rcsvr.pkl -O row50_8_reduced -f topic -LDA row50_8_reduced

cd $BASEPATH/model
#python pred_sherlock.py -c $BASEPATH/configs/pred_sherlock_config.txt
#python sherlock_exp.py -c $BASEPATH/configs/cv_sherlock_config.txt
#python sherlock_exp.py -c $BASEPATH/configs/cv_sherlock_topic_config.txt
#python sherlock_exp.py -c $BASEPATH/configs/cv_sherlock_config.txt
python CRF_exp.py -c $BASEPATH/configs/cv_CRF_sherlock_config.txt