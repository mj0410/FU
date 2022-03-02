export BASEPATH=C:/Users/minie/Desktop/git/MS_THESIS/scripts_draft/Sato
export INPUT_DIR=C:/Users/minie/Desktop/git/MS_THESIS/test_data
export SHERLOCKPATH=$BASEPATH/sherlock
export EXTRACTPATH=$BASEPATH/extract
export PYTHONPATH=$PYTHONPATH:$SHERLOCKPATH
export PYTHONPATH=$PYTHONPATH:$BASEPATH
export LDANAME='t18_rr'

export TYPENAME='synthea_rr'
cd $BASEPATH/model
python pred_CRF.py -c $BASEPATH/configs/pred_CRF_config.txt

#export TYPENAME='reduced_synthea'
#cd $BASEPATH/model
#python pred_CRF.py -c $BASEPATH/configs/pred_CRF_config.txt