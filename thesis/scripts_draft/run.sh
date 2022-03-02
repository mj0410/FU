export ORIGIN=C:/Users/minie/Desktop/git/MS_THESIS/scripts_draft
export BASEPATH=C:/Users/minie/Desktop/git/MS_THESIS/scripts_draft/Sato

export DT_MODEL_DIR=$ORIGIN/DeepTable/models/test_new
#export INPUT_DIR=C:/Users/minie/Desktop/git/MS_THESIS/test_data_woDT
export INPUT_DIR=C:/Users/minie/Desktop/FU/Thesis/Synthea/100/csv

export SHERLOCKPATH=$BASEPATH/sherlock
export EXTRACTPATH=$BASEPATH/extract
export PYTHONPATH=$PYTHONPATH:$SHERLOCKPATH
export PYTHONPATH=$PYTHONPATH:$BASEPATH
export TYPENAME='syntheaA'
export LDANAME='t18'
export DTOUTPUT='synthea100_csv'

#cd $BASEPATH/model
#python pred_CRF.py -i $INPUT_DIR -n 50 -t $LDANAME -m CRF_t18_50-fold -o CRF_t18_100p_woDT

#cd $ORIGIN/DeepTable
#python DeepTablePred.py -m $DT_MODEL_DIR/model_100.hdf5 -i $INPUT_DIR -o $DTOUTPUT

while true
do
  echo "Continue to semantic type detection YES[y] NO[n]"
  read ans

  if [[ "$ans" == "y" ]] || [[ "$ans" == "yes" ]]; then
    cd $BASEPATH/model
    python pred_CRF.py -dt $DTOUTPUT -n 50 -m CRF_t18_50-fold -o CRF_t18_100_csv -t $LDANAME
    break
  elif [[ "$ans" == "n" ]] || [[ "$ans" == "no" ]]; then
    echo "quit"
    break
  else
    echo "please enter yes/y or no/n"
  fi
done

#cd $BASEPATH/model
#python pred_sherlock.py -c $BASEPATH/configs/pred_sherlock_config.txt
#python pred_CRF.py -c $BASEPATH/configs/pred_CRF_config.txt

