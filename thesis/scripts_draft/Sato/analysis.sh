export BASEPATH=C:/Users/minie/Desktop/git/MS_THESIS/scripts_draft/Sato
export INPUT_DIR=C:/Users/minie/Desktop/git/MS_THESIS/DBs/Synthea/1K
export PYTHONPATH=$PYTHONPATH:$SHERLOCKPATH
export PYTHONPATH=$PYTHONPATH:$BASEPATH
export RESULT=C:/Users/minie/Desktop/results/new_data

export TYPENAME='syntheaA'
#python result_analysis.py --split split_50fold_syntheaA --mode num --num_name A

export sherlock_model=sherlock
export topic_model1=t9
export topic_model2=t18

export crf_sherlock_model=CRF_sherlock
export crf_topic_model1=CRF_topic9
export crf_topic_model2=CRF_topic18

export topic1_num=9
export topic2_num=18

python result_analysis.py --mode avg

export topic_model2=sherlock
export crf_topic_model2=CRF_sherlock
python result_analysis.py --mode violin2 --violinplot_name sherlock

#export topic_model2=t18
#export crf_topic_model2=CRF_t18
#python result_analysis.py --mode violin2 --violinplot_name t9
#python result_analysis.py --mode bar --barplot_name t18

export TYPENAME='syntheaB'

#export topic_model2=t18
#export crf_topic_model2=CRF_rcol_t18
#python result_analysis.py --mode violin2 --violinplot_name t9
#python result_analysis.py --mode bar --barplot_name t18

#export sherlock_model=t18
#export topic_model1=synthea_row50_100exp_topic9
#export topic1_num=9
#export topic_model2=synthea_row50_100exp_topic18
#export topic2_num=18

#python result_analysis.py --split split_row50_1000fold_synthea --mode num --num_name ori
#python result_analysis.py --split split_row50_rcsv_1000fold_synthea_local --mode num --num_name rcsv
#python result_analysis.py --mode violin3 --violinplot_name csv_sherlock
#python result_analysis.py --mode bar --barplot_name csv_sherlock

#export sherlock_model=synthea_row50_100exp_sherlock_rcsv
#export topic_model1=synthea_row50_100exp_topic9_rcsv
#export topic1_num=9
#export topic_model2=synthea_row50_100exp_topic17_rcsv
#export topic2_num=17

#python result_analysis.py --mode violin3 --violinplot_name rcsv_sherlock
#python result_analysis.py --mode bar --barplot_name rcsv_sherlock

#export crf_sherlock_model=CRF_t18
#export crf_topic_model1=CRF_topic9
#export topic1_num=9
#export crf_topic_model2=CRF_topic18
#export topic2_num=18

#python result_analysis.py --mode violin3 --violinplot_name CRF
#python result_analysis.py --mode bar --barplot_name CRF

#export crf_sherlock_model=sherlock_rcsv
#export crf_topic_model1=topic9_rcsv
#export topic1_num=9
#export crf_topic_model2=topic17_rcsv
#export topic2_num=17

#python result_analysis.py --mode violin3 --violinplot_name rcsv_CRF
#python result_analysis.py --mode bar --barplot_name rcsv_CRF

#python result_analysis.py --mode violin2 --violinplot_name s_rcsv

export TYPENAME='syntheaB'
#python result_analysis.py --split split_50fold_syntheaB --mode num --num_name B

#export topic_model2=t18
#export crf_topic_model2=CRF_rcol_t18
#python result_analysis.py --mode bar --barplot_name t18

export sherlock_model=sherlock
export crf_sherlock_model=sherlock
#python result_analysis.py --mode violin2 --violinplot_name sherlock

export sherlock_model=t18
export crf_sherlock_model=CRF_rcol_t18
#python result_analysis.py --mode violin2 --violinplot_name t18

#export sherlock_model=reduced_synthea_row50_100exp_sherlock
#export topic_model1=reduced_synthea_row50_100exp_topic9
#export topic1_num=9
#export topic_model2=reduced_synthea_row50_100exp_topic18
#export topic2_num=18

#python result_analysis.py --split split_row50_1000fold_reduced_synthea --mode num --num_name ori
#python result_analysis.py --split split_row50_rcsv_1000fold_reduced_synthea_local --mode num --num_name rcsv

#python result_analysis.py --mode violin3 --violinplot_name sherlock
#python result_analysis.py --mode bar --barplot_name sherlock

#export sherlock_model=reduced_synthea_row50_100exp_sherlock_rcsv
#export topic_model1=reduced_synthea_row50_100exp_topic9_rcsv
#export topic1_num=9
#export topic_model2=reduced_synthea_row50_100exp_topic17_rcsv
#export topic2_num=17

#python result_analysis.py --mode violin3 --violinplot_name rcsv_sherlock
#python result_analysis.py --mode bar --barplot_name rcsv_sherlock

#export crf_sherlock_model=reduced_CRF
#export crf_topic_model1=CRF_reduced_topic9
#export topic1_num=9
#export crf_topic_model2=CRF_reduced_topic18
#export topic2_num=18

#python result_analysis.py --mode violin3 --violinplot_name CRF
#python result_analysis.py --mode bar --barplot_name CRF

#export crf_sherlock_model=reduced_sherlock_rcsv
#export crf_topic_model1=topic9_reduced_rcsv
#export topic1_num=9
#export crf_topic_model2=topic17_reduced_rcsv
#export topic2_num=17

#python result_analysis.py --mode violin3 --violinplot_name rcsv_CRF
#python result_analysis.py --mode bar --barplot_name rcsv_CRF

#python result_analysis.py --mode violin2 --violinplot_name s_rcsv
