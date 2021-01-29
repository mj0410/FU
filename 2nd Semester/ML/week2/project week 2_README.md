# Week2 - [Video Report ](https://voicethread.com/share/14370306/)

>[![ForTheBadge built-by-developers](http://ForTheBadge.com/images/badges/built-by-developers.svg)](https://GitHub.com/Naereen/)  🤖 TeamE 🤖


----
## FitHic2?
FitHic2 compute statistical confidence estimates for Hi-C to identify significant chromatin contacts
genome-wide analysis for high-resolution Hi-C data, including all intra- chromosomal distances and inter-chromosomal contacts

see [FitHic2-github](https://ay-lab.github.io/fithic/)

see [FitHic2-paper](https://www.nature.com/articles/s41596-019-0273-0/t)


## Installation - Windows
1. Get Ubuntu from  Microsoft Store
2. Prepare Environment

```bash
##-python 3-
sudo apt-get update
sudo apt install python3-pip
pip3 install networkx
pip3 install wheel
pip3 install fithic
##-python 2- 
sudo apt install python-minimal
sudo apt install python-pip
python2 -m pip install numpy
```
3. Installing software
```bash
git clone https://github.com/ay-lab/fithic.git
cd fithic/fithic
FITHICDIR=$(pwd)
```

4. **Alles Gut!**

**Follow more installation instruction  please visit the paper or github (https://ay-lab.github.io/fithic/)**

**For Mac and Linux system please visit https://ay-lab.github.io/fithic/**


## How-to-Run

### Prepare Data

```bash
wget http://fithic.lji.org/fithic_protocol_data.tar.gz
tar -xvzf fithic_protocol_data.tar.gz
cd fithic_protocol_data/data
DATADIR=$(pwd)
```

#### A brief overview of the main stages of analysis performed by FitHiC2
![a](https://media.springernature.com/full/springer-static/image/art%3A10.1038%2Fs41596-019-0273-0/MediaObjects/41596_2019_273_Fig1_HTML.png)

#### Flowchart of FitHiC2 parameter and configuration setting choices.
![b](https://media.springernature.com/full/springer-static/image/art%3A10.1038%2Fs41596-019-0273-0/MediaObjects/41596_2019_273_Fig3_HTML.png)



### Step 1-4 - Generation of input files for FitHiC2

```bash
## To first generate the contact maps file
bash $FITHICDIR/utils/validPairs2FitHiC-fixedSize.sh 40000 IMR90
$DATADIR/validPairs/IMR90_HindIII_r4.hg19.bwt2pairs.withSingles.
mapq30.validPairs.gz $DATADIR/contactCounts

## To generate the second input file for FitHiC2, the fragment mappability
python3 $FITHICDIR/utils/createFitHiCFragments-fixedsize.py
--chrLens $DATADIR/referenceGenomes/hg19wY-lengths
--resolution 40000
--outFile $DATADIR/fragmentMappability/IMR90_fithic.fragmentsfile.gz

## To generate the fragment mappability file, but for a non-fixed-size dataset,
bash $FITHICDIR/utils/createFitHiCFragments-nonfixedsize.sh $DATADIR/
fragmentMappability/yeast_fithic.fragments HindIII $DATADIR/reference
Genomes/yeast_reference_sequence_R62-1-1_20090218.fsa

## Computing biases
python3 $FITHICDIR/utils/HiCKRy.py
-i $DATADIR/contactCounts/Duan_yeast_EcoRI.gz
-f $DATADIR/fragmentMappability/Duan_yeast_EcoRI.gz
-o $DATADIR/biasValues/Duan_yeast_EcoRI.gz

python3 $FITHICDIR/utils/HiCKRy.py
-i $DATADIR/contactCounts/Dixon_IMR90-wholegen_40kb.gz
-f $DATADIR/fragmentMappability/Dixon_IMR90-wholegen_40kb.gz
-o $DATADIR/biasValues/Dixon_IMR90-wholegen_40kb.gz

python3 $FITHICDIR/utils/HiCKRy.py
-i $DATADIR/contactCounts/Rao_GM12878-primary-chr5_5kb.gz
-f $DATADIR/fragmentMappability/Rao_GM12878-primary-
```

### Step 5-6 - Running FitHiC2: 5-kb chromosome 5 human

```bash
python3 $FITHICDIR/fithic.py
-i $DATADIR/contactCounts/Rao_GM12878-primary-chr5_5kb.gz
-f $DATADIR/fragmentMappability/Rao_GM12878-primary-chr5_5kb.gz
-t $DATADIR/biasValues/Rao_GM12878-primary-chr5_5kb.gz
-r 5000
-o $DATADIR/fithicOutput/Rao_GM12878-primary-chr5_5kb
-l Rao_GM12878-primary-chr5_5kb
-U 1000000
-L 15000
-v
```

### Step 7 - To create an HTML report summarizing

```bash
bash $FITHICDIR/utils/createFitHiCHTMLout.sh Rao_GM12878-primarychr5_5kb 1 $DATADIR/fithicOutput/Rao_GM12878-primary-chr5_5kb

```
### Step 8 -Running FitHiC2: yeast
```bash
python3 $FITHICDIR/fithic.py
-i $DATADIR/contactCounts/Duan_yeast_EcoRI.gz
-f $DATADIR/fragmentMappability/Duan_yeast_EcoRI.gz
-t $DATADIR/biasValues/Duan_yeast_EcoRI.gz
-r 0

```

**For more example see [FitHic2-paper](https://www.nature.com/articles/s41596-019-0273-0/t)**

----

## FitHiC2 - Windows Troubleshooting

1. For windows, windows style characters(e.g. "\r") can cause issues in cygwin. style .
    Using Notepad++ , you can correct the specific file using the option 
     
    *  Edit tab-> select EOL conversion -> Unix/OSX format -> save file


**For original troubleshooting advice  https://www.nature.com/articles/s41596-019-0273-0/tables/8**

## Changelog
* 5-5-2020 first submit


<span style="font-family:Papyrus; font-size:4em;">LOVE!</span>
