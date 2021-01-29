# Week8 - [Video Report](https://voicethread.com/share/14713170/)

>[![ForTheBadge built-by-developers](http://ForTheBadge.com/images/badges/built-by-developers.svg)](https://GitHub.com/Naereen/)  🤖 TeamE 🤖

----

## How To Run

1. Install PyMethylProcess [PyMethylProcess](https://github.com/Christensen-Lab-Dartmouth/PyMethylProcess)
2. docker pull joshualevy44/pymethylprocess:0.1.3
3. Follow process from [PyMethylProcess_wiki](https://github.com/Christensen-Lab-Dartmouth/PyMethylProcess/wiki)
   
We use GSE108576 data from [GSE](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE108576)

```bash
docker run -it joshualevy44/pymethylprocess:0.1.3

##download data
pymethyl-preprocess download_geo -g GSE108576 -o geo_idats/

##formatting sample sheet
nano include_col.txt

gender:ch1	Sex
description.1	ID

pymethyl-preprocess create_sample_sheet -is geo_idats/GSE108576_clinical_info.csv -s geo -i geo_idats/ -os geo_idats/samplesheet.csv -d "disease state:ch1" -c include_col.txt
mkdir backup_clinical && mv geo_idats/GSE108576_clinical_info.csv backup_clinical
pymethyl-preprocess meffil_encode -is geo_idats/samplesheet.csv -os geo_idats/samplesheet.csv

##preprocessing pipeline
pymethyl-preprocess preprocess_pipeline -i geo_idats/ -p minfi -noob -qc                          
pymethyl-preprocess preprocess_pipeline -i geo_idats/ -p minfi -noob -u -n 2

##final preprocessing step
pymethyl-utils remove_sex -i preprocess_outputs/methyl_array.pkl
pymethyl-preprocess na_report -i autosomal/methyl_array.pkl -o na_report/
pymethyl-preprocess imputation_pipeline -i ./autosomal/methyl_array.pkl -s sklearn -m MICE -st 0.05 -ct 0.05

##feature selection
pymethyl-preprocess feature_select -n 55

##split the data
pymethyl-utils train_test_val_split -tp .8 -vp .125

```

4. run classifiers(ML_week8.ipynb) using generated methylation arrays (train, validation, test)
5. run MethylNet using generated methylation arrays [MethylNet](https://github.com/Christensen-Lab-Dartmouth/MethylNet)

```bash
methylnet-embed launch_hyperparameter_scan -sc disease -mc 0.84 -b 1. -g -j 10
methylnet-embed launch_hyperparameter_scan -sc disease -g -n 1

methylnet-predict launch_hyperparameter_scan -ic disease -cat -g -mc 0.84 -j 10
methylnet-predict launch_hyperparameter_scan -ic disease -cat -g -n 1

methylnet-predict make_prediction -cat

```

6. compare the results to original paper [paper](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6219520/pdf/41467_2018_Article_6715.pdf)

<span style="font-family:Papyrus; font-size:4em;">LOVE :heart:  :tw: :kr: :th: :cn: !</span>
