# Week3 - [Video Report ]()

>[![ForTheBadge built-by-developers](http://ForTheBadge.com/images/badges/built-by-developers.svg)](https://GitHub.com/Naereen/)  

🤖 TeamE 🤖 

----
This week’s task is about the detection of somatic mutations from genomic sequences with a
software called NeuSomatic – which is, at its core, a convolutional neural network.

Tasks:
* Perform an analysis on real data
* Perform an analysis on synthetic data

## Background Knowledge

### CNN architecture

- [Comprehensive-guide-to-convolutional-neural-networks](https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53)
- [Batch normalization](https://towardsdatascience.com/batch-normalization-in-neural-networks-1ac91516821c)
- [Softmax Function ](https://medium.com/data-science-bootcamp/understand-the-softmax-function-in-minutes-f3a59641e86d)
- [Multi-Class Neural Networks: Softmax](https://developers.google.com/machine-learning/crash-course/multi-class-neural-networks/softmax)
- More CNN variants: [Click me](https://towardsdatascience.com/illustrated-10-cnn-architectures-95d78ace614d) / [Click me](https://www.jeremyjordan.me/convnet-architectures/)

### NeuSomatic

NeuSomatic, the convolutional neural network approach for somatic mutation detection.

**A.Input mutation matrix**;
3-dimensional matrix M with k channels of size 5 × 32. The five rows in each channel corresponds to four DNA bases A, C, G, and T, and the gap character (‘−’). Each of the 32 columns of the matrix represents one column of the alignment. 

In channel;
* C1 = reference base
* C2 = tumor reads
* C3 = normal reads
* from C4 to Ck = alignment propetries  such as coverage, base quality, mapping quality, strands, clipping information, edit distance, alignment score, and paired-end information

**B.CNN architecture**;
nine convolutional layers
*  The input matrices are fed into the first convolution layer with 64 output channels
*  First convolution layer with 64 output channels 1 × 3 kernel size and Relu activation followed by a batch normalization and a max-pooling layer
*  Four blocks with shortcut identity connection similar to ResNet structure.These blocks consist of a convolution layer with 3 × 3 kernels followed by batch normalization and a convolution layer with 5 × 5 kernels. Between these shortcut blocks, they use batch normalization and max-pooling layers
* The output of final block is fed to a fully connected layer of size 240.
* The resulting feature vector is then fed to two softmax classifiers and a regressor.

![Image](https://media.springernature.com/full/springer-static/image/art%3A10.1038%2Fs41467-019-09027-x/MediaObjects/41467_2019_9027_Fig1_HTML.png)

* Results: The first classifier is a 4-way classifier that predicts the mutation type from the four classes of non-somatic, SNV, insertion, and deletion. The second classifier predicts the length of the predicted mutation from the four categories of 0, 1, 2, and ≥3. Non-somatic calls are annotated as zero size mutations, SNVs and 1-base INDELs are annotated as size 1, while 2-base and ≥3 size INDELs are, respectively, annotated as 2 and ≥3 size mutations. The regressor predicts the column of the mutations in the matrix, to assure the prediction is targeted the right position and is optimized using a smooth L1 loss function.

**Reference**: [Read more NeuSomatic](https://media.springernature.com/full/springer-static/image/art%3A10.1038%2Fs41467-019-09027-x/MediaObjects/41467_2019_9027_Fig1_HTML.png)

### Some questions you should be able to answer

* What are somatic mutations?
> An alteration in DNA that occurs after conception. Somatic mutations can occur in any of the cells of the body except the germ cells (sperm and egg) and therefore are not passed on to children. These alterations can (but do not always) cause cancer or other diseases. 
> [NCI Dictionary of Cancer Terms](https://www.cancer.gov/publications/dictionaries/cancer-terms/def/somatic-mutation)

* What is the input to the NeuSomatic algorithm?
> For training mode:
>
> * tumor .bam alignment file
> * normal .bam alignment file
> * training region .bed file
> * truth somatic variant .vcf file
>
> For calling mode:
>
> * tumor .bam alignment file
> * normal .bam alignment file
> * call region .bed file
> * trained model .pth file

* Why do you need tumor reads AND normal reads?
>
* How are the input sequences aligned? (What is multiple sequence alignment?)
>
* What is the output of the NeuSomatic algorithm?
> Variant call format (vcf) file, which contains information about each variants predicted by NeuSomatic

* What is the F1-score?
> ![image](https://miro.medium.com/max/1400/1*OhEnS-T54Cz0YSTl_c3Dwg.jpeg)
> 
> Based on confusion matrix, we can calculate precision and recall;
>
> ![Image](https://miro.medium.com/max/888/1*7J08ekAwupLBegeUI8muHA.png)
>
> precision = how many of predicted positive are actual positive
> recall = how many of actual positive are predicted as positive
>
> ![image](https://miro.medium.com/max/564/1*T6kVUKxG_Z4V5Fm1UXhEIw.png)
>
> F1 score is harmonic mean between precision and recall, used as statistical measure to rate performance.

* What is PacBio data?
> 

* What are INDELs?
> an insertion or deletion of bases in the genome

* What is sequence coverage?
>
* How is the synthetic data created in the paper? What are the main assumptions to create this kind of data?
>
* What is a ResNet?
> A residual neural network is a Convolutional Neural Network (CNN)  architecture which was designed to enable hundreds or thousands of convolutional layers. ResNet do this by utilizing skip connections, or shortcuts to jump over some layers to avoid the problem of vanishing gradients.
-More information [Click me](https://towardsdatascience.com/introduction-to-resnets-c0a830a288a4)

* What are hyper-parameters?
>



## Download data and install the software

### 1. NeuSomatic software.

Most of the installation instuction already described in [NeuSomatic](https://github.com/bioinform/neusomatic) 
However, we faced several required libraries in order to install software properly.  
```bash 
sudo apt-get install libz-dev
sudo apt-get install libbz2-dev
sudo apt-get install liblzma-dev
sudo apt-get install libcurl4-openssl-dev
sudo apt-get install libncurses5-dev libncursesw5-dev
sudo apt install autoconf
sudo apt install unzip
sudo apt-get install -y pkg-config
sudo apt-get install texinfo
sudo apt-get install libglib2.0-dev
```
**[Achtung!!:warning:] Windows Subsystem for Linux**

If you want to use docker version of NeuSomatic There's something you need to understand first. The Docker Engine does not run on WSL, you have to have "Docker For Windows" installed on your host machine. Next, follow [Installing the Docker client on Windows Subsystem for Linux (Ubuntu)](https://medium.com/@sebagomez/installing-the-docker-client-on-ubuntus-windows-subsystem-for-linux-612b392a44c4) to make it runnable.



### 2. Synthetic data - software.

**BamSurgeon** - [How-To-Install](https://github.com/adamewing/bamsurgeon)

> [Achtung!!:warning:] Don't forget to create directory and export; command to do ->  `mkdir $HOME/bin | export PATH="$HOME/bin:$PATH"`

You may also want to see [Dockerized in silico somatic mutation spike in pipeline to generate training data set with ground truths](https://github.com/bioinform/somaticseq/tree/master/utilities/dockered_pipelines/bamSimulator) for workflow to create a synthetic dataset.


### 3. Data

For this analysis we use 

1. HCC1143 (breast cancer) -> [Information](https://www.wikidata.org/wiki/Q54881530)  / [BAM format]( https://console.cloud.google.com/storage/browser/gatk-best-practices/somatic-hg38?project=broad-dsde-outreach )
```
## For  Chromosme 21 region arround 19000000-20000000
http://content.cruk.cam.ac.uk/bioinformatics/CourseData/IntroToIGV/HCC1143.normal.21.19M-20M.bam
http://content.cruk.cam.ac.uk/bioinformatics/CourseData/IntroToIGV/HCC1143.tumour.21.19M-20M.bam
```

2. Test samples under test directory of each softwares 
3. Reference file `Homo_sapiens_assembly19.fasta` , however you can use any [version](https://gatk.broadinstitute.org/hc/en-us/articles/360035890711-GRCh37-hg19-b37-humanG1Kv37-Human-Reference-Discrepancies) you want.


## Perform an analysis on real data

### Identify variants 

To prepare reference file 
> `samtools faidx Homo_sapiens_assembly19.fasta`

To generate `.bed` file , you can view at `Homo_sapiens_assembly19.fasta.fai`

or
```
samtools faidx $fasta
awk 'BEGIN {FS="\t"}; {print $1 FS "0" FS $2}' $fasta.fai > $fasta.bed
```

For format description- [Bed Format](https://genome.ucsc.edu/FAQ/FAQformat.html#format1)


1. Peprocess step in call mode (scan the alignments, find candidates, generate input matrix dataset)

```bash
python neusomatic/neusomatic/python/preprocess.py \
	--mode call \
	--reference /mnt/c/MasterBioinfomatics/Semester2/ML/week3/datasets/Homo_sapiens_assembly19.fasta \
	--region_bed /mnt/c/MasterBioinfomatics/Semester2/ML/week3/datasets/refs/genome.chr21.bed \
	--tumor_bam  /mnt/c/MasterBioinfomatics/Semester2/ML/week3/datasets/HCC1143.tumor.21.19M-20M.bam \
	--normal_bam /mnt/c/MasterBioinfomatics/Semester2/ML/week3/datasets/HCC1143.normal.21.19M-20M.bam \
	--work work_call \
	--min_mapq 10 \
	--num_threads 4 \
	--scan_alignments_binary neusomatic/neusomatic/bin/scan_alignments
```

**[Achtung!!:warning:]**
Be careful at reference genome version and alignment map  (BAM file)

2. Call variants

```
CUDA_VISIBLE_DEVICES=0 python neusomatic/neusomatic/python/call.py \
	--candidates_tsv work_call/dataset/*/candidates*.tsv \
	--reference /mnt/c/MasterBioinfomatics/Semester2/ML/week3/datasets/Homo_sapiens_assembly19.fasta \
	--out work_call \
	--checkpoint neusomatic/neusomatic/models/NeuSomatic_v0.1.4_standalone_SEQC-WGS-Spike.pth \
	--num_threads 4 \
	--batch_size 100
```

3. Postprocess step (resolve long INDEL sequences, report vcf)
```
python neusomatic/neusomatic/python/postprocess.py \
	--reference /mnt/c/MasterBioinfomatics/Semester2/ML/week3/datasets/Homo_sapiens_assembly19.fasta \
	--tumor_bam /mnt/c/MasterBioinfomatics/Semester2/ML/week3/datasets/HCC1143.tumor.21.19M-20M.bam \
	--pred_vcf work_call/pred.vcf \
	--candidates_vcf work_call/work_tumor/filtered_candidates.vcf \
	--output_vcf work_call/HCC1143.tumor.21.19M-20M.NeuSomatic.vcf \
	--work work_call 
```

## Perform an analysis on synthetic data
This task we will use Bamsurgeon- [Manual-parameter.pdf](https://github.com/adamewing/bamsurgeon/blob/master/doc/Manual.pdf) , you can also use [somaticseq-bamSimulator pipline](https://github.com/bioinform/somaticseq/tree/master/utilities/dockered_pipelines/bamSimulator) to create synthetic data

### A. Create synthetic data

1. Add SNV to bam file 
```
bamsurgeon/bin/addsnv.py --snvfrac 0.1 --mutfrac 0.5 --coverdiff 0.9 --procs 1 \
--varfile /mnt/c/MasterBioinfomatics/Semester2/ML/week3/bamsurgeon/test_data/random_snvs.txt \
--bamfile /mnt/c/MasterBioinfomatics/Semester2/ML/week3/bamsurgeon/test_data/testregion_realign.bam \
--reference /mnt/c/MasterBioinfomatics/Semester2/ML/week3/bamsurgeon/test_data/Homo_sapiens_chr22_assembly19.fasta \
--outbam /mnt/c/MasterBioinfomatics/Semester2/ML/week3/BSdata/unsorted.snvs.added.bam \
--tmpdir /mnt/c/MasterBioinfomatics/Semester2/ML/week3/BSdata \
--mindepth 5 --maxdepth 5000 --minmutreads 2 --seed 12417 \
--picardjar /mnt/c/MasterBioinfomatics/Semester2/ML/week3/tools/picard-tools-1.131/picard.jar  \
--force  --aligner  mem
```

2.  Add INDEL to bam file(Optional)
```
bamsurgeon/bin/addindel.py \
--snvfrac 0.1 --mutfrac 0.5 --coverdiff 0.9 --procs 1 \
--varfile /mnt/c/MasterBioinfomatics/Semester2/ML/week3/bamsurgeon/test_data/test_indels.txt \
--bamfile /mnt/c/MasterBioinfomatics/Semester2/ML/week3/bamsurgeon/test_data/unsorted.snvs.added.bam \
--reference /mnt/c/MasterBioinfomatics/Semester2/ML/week3/bamsurgeon/test_data/Homo_sapiens_chr22_assembly19.fasta \
--outbam /mnt/c/MasterBioinfomatics/Semester2/ML/week3/Syntheticdata/unsorted.snvs.indels.added.bam \
--picardjar /mnt/c/MasterBioinfomatics/Semester2/ML/week3/tools/picard-tools-1.131/picard.jar \
--tmpdir /mnt/c/MasterBioinfomatics/Semester2/ML/week3/BSdata \
--mindepth 5 \
--maxdepth 5000 \
--minmutreads 2 \
--seed 12417 \
--tagreads --force \
--aligner mem
```

Sort File 
```
samtools sort unsorted.snvs.indels.added.bam > tumor.sorted.testBS.bam
samtools index tumor.sorted.testBS.bam
```

### B. Perform training on synthetic data

For truth_vcf we rename `mv unsorted.snvs.indels.added.vcf >  truth.testBS.vcf `

1. Preprocess step in train mode (scan the alignments, find candidates, generate input matrix dataset)
```
python neusomatic/neusomatic/python/preprocess.py \
	--mode train \
	--reference /mnt/c/MasterBioinfomatics/Semester2/ML/week3/bamsurgeon/test_data/Homo_sapiens_chr22_assembly19.fasta  \
	--region_bed /mnt/c/MasterBioinfomatics/Semester2/ML/week3/datasets/genome.chr22.bed  \
	--tumor_bam /mnt/c/MasterBioinfomatics/Semester2/ML/week3/BSdata/tumor.sorted.testBS.bam \
	--normal_bam /mnt/c/MasterBioinfomatics/Semester2/ML/week3/BSdata/normal.testBS.bam \
	--work /mnt/c/MasterBioinfomatics/Semester2/ML/week3/BSdata/work_train \
	--truth_vcf /mnt/c/MasterBioinfomatics/Semester2/ML/week3/BSdata/truth.testBS.vcf \
	--min_mapq 10 \
	--num_threads 2 \
	--scan_alignments_binary neusomatic/neusomatic/bin/scan_alignments
```

2.Train network
```
CUDA_VISIBLE_DEVICES=0,1 python  neusomatic/neusomatic/python/train.py \
	--candidates_tsv /mnt/c/MasterBioinfomatics/Semester2/ML/week3/BSdata/work_train/dataset/*/candidates*.tsv \
	--out /mnt/c/MasterBioinfomatics/Semester2/ML/week3/BSdata/work_train \
	--num_threads 10 \
	--batch_size 100 
```

### C. Call mutation 

1. Preprocess step in call mode (scan the alignments, find candidates, generate input matrix dataset)
```
python neusomatic/neusomatic/python/preprocess.py \
	--mode call \
	--reference /mnt/c/MasterBioinfomatics/Semester2/ML/week3/bamsurgeon/test_data/Homo_sapiens_chr22_assembly19.fasta \
	--region_bed /mnt/c/MasterBioinfomatics/Semester2/ML/week3/datasets/genome.chr22.bed \
	--tumor_bam   /mnt/c/MasterBioinfomatics/Semester2/ML/week3/BSdata/tumor.sorted.testBS.bam  \
	--normal_bam /mnt/c/MasterBioinfomatics/Semester2/ML/week3/BSdata/normal.testBS.bam \
	--work /mnt/c/MasterBioinfomatics/Semester2/ML/week3/BSdata/work_call \
	--min_mapq 10 \
	--num_threads 4 \
	--scan_alignments_binary neusomatic/neusomatic/bin/scan_alignments
```
2. Call variants
```
CUDA_VISIBLE_DEVICES=0 python neusomatic/neusomatic/python/call.py \
	--candidates_tsv /mnt/c/MasterBioinfomatics/Semester2/ML/week3/BSdata/work_call/dataset/*/candidates*.tsv \
	--reference /mnt/c/MasterBioinfomatics/Semester2/ML/week3/bamsurgeon/test_data/Homo_sapiens_chr22_assembly19.fasta \
	--out /mnt/c/MasterBioinfomatics/Semester2/ML/week3/BSdata/work_call \
	--checkpoint /mnt/c/MasterBioinfomatics/Semester2/ML/week3/BSdata/work_train/models/checkpoint_neusomatic_20-05-08-23-52-53_epoch1000.pth \
	--num_threads 4 \
	--batch_size 100
```
3. Postprocess step (resolve long INDEL sequences, report vcf)
```
python neusomatic/neusomatic/python/postprocess.py \
	--reference /mnt/c/MasterBioinfomatics/Semester2/ML/week3/bamsurgeon/test_data/Homo_sapiens_chr22_assembly19.fasta \
	--tumor_bam /mnt/c/MasterBioinfomatics/Semester2/ML/week3/BSdata/tumor.sorted.testBS.bam \
	--pred_vcf /mnt/c/MasterBioinfomatics/Semester2/ML/week3/BSdata/work_call/pred.vcf \
	--candidates_vcf /mnt/c/MasterBioinfomatics/Semester2/ML/week3/BSdata/work_call/work_tumor/filtered_candidates.vcf \
	--output_vcf /mnt/c/MasterBioinfomatics/Semester2/ML/week3/BSdata/work_call/tumor.sorted.testBS.NeuSomatic.vcf \
	--work /mnt/c/MasterBioinfomatics/Semester2/ML/week3/BSdata/work_call 

```

## Evaluate the results 

### For Perform an analysis on real data

We  download [Control.vcf](https://dcc.icgc.org/releases/PCAWG/cell_lines/HCC1143)(519b8381-95d5-4fce-a90c-7576cce2110c.dkfz-snvCalling_1-0-132-1.20160126.somatic.snv_mnv.vcf.gz) file and compare the called mutaiton file.

**HCC1143.tumor.21.19M-20M.NeuSomatic.vcf vs. Control.vcf** 
```
## HCC1143.tumor.21.19M-20M.NeuSomatic.vcf
21	19024087	.	C	A	12.6022	PASS	SCORE=0.9451;DP=157;RO=139;AO=18;AF=0.1146	GT:DP:RO:AO:AF	0/1:157:139:18:0.1146
21	19034179	.	A	G	33.9806	PASS	SCORE=0.9996;DP=88;RO=54;AO=33;AF=0.3793	GT:DP:RO:AO:AF	0/1:88:54:33:0.3793
21	19046286	.	A	T	35.2289	PASS	SCORE=0.9997;DP=84;RO=72;AO=12;AF=0.1429	GT:DP:RO:AO:AF	0/1:84:72:12:0.1429
21	19207689	.	C	G	12.2415	PASS	SCORE=0.9403;DP=165;RO=149;AO=16;AF=0.097	GT:DP:RO:AO:AF	0/1:165:149:16:0.097
21	19316282	.	C	T	100.0000	PASS	SCORE=1.0000;DP=213;RO=124;AO=89;AF=0.4178	GT:DP:RO:AO:AF	0/1:213:124:89:0.4178
21	19405970	.	GGCC	G	1.6702	REJECT	SCORE=0.3193;DP=46;RO=45;AO=1;AF=0.0217	GT:DP:RO:AO:AF	0/1:46:45:1:0.0217
21	19414361	.	C	G	14.8243	PASS	SCORE=0.9671;DP=24;RO=20;AO=4;AF=0.1667	GT:DP:RO:AO:AF	0/1:24:20:4:0.1667
21	19472161	.	G	A	39.9993	PASS	SCORE=0.9999;DP=47;RO=36;AO=11;AF=0.234	GT:DP:RO:AO:AF	0/1:47:36:11:0.234
21	19499226	.	A	C	33.0111	PASS	SCORE=0.9995;DP=47;RO=38;AO=8;AF=0.1739	GT:DP:RO:AO:AF	0/1:47:38:8:0.1739
21	19544778	.	A	T	39.9993	PASS	SCORE=0.9999;DP=62;RO=42;AO=20;AF=0.3226	GT:DP:RO:AO:AF	0/1:62:42:20:0.3226
21	19866637	.	G	T	33.9806	PASS	SCORE=0.9996;DP=35;RO=22;AO=13;AF=0.3714	GT:DP:RO:AO:AF	0/1:35:22:13:0.3714
21	19873456	.	A	G	28.2404	PASS	SCORE=0.9985;DP=46;RO=38;AO=8;AF=0.1739	GT:DP:RO:AO:AF	0/1:46:38:8:0.1739
21	19959492	.	CCT	C	3.8874	LowQual	SCORE=0.5914;DP=40;RO=36;AO=4;AF=0.1	GT:DP:RO:AO:AF	0/1:40:36:4:0.1

## Control.vcf
21	19003680	.	A	G	.	TAC;FRQ;1PS	SOMATIC;SNP;AF=0.00,0.02;MQ=60	GT:DP:DP4	0/0:39:16,23,0,0	0/0:103:52,49,1,1
21	19008875	.	A	G	.	TAC;FRQ;1PS	SOMATIC;SNP;AF=0.00,0.02;MQ=60	GT:DP:DP4	0/0:59:35,24,0,0	0/0:103:56,45,1,1
21	19024087	.	C	A	.	PASS	SOMATIC;SNP;AF=0.00,0.10;MQ=60	GT:DP:DP4	0/0:61:31,30,0,0	0/0:120:46,62,3,9
21	19024251	.	C	A	.	RE;TAC;SBAF;FRQ	SOMATIC;SNP;AF=0.00,0.02;MQ=60	GT:DP:DP4	0/0:40:6,34,0,0	0/0:103:21,80,0,2
21	19034179	.	A	G	.	PASS	SOMATIC;SNP;AF=0.00,0.37;MQ=60	GT:DP:DP4	0/0:29:12,17,0,0	0/1:67:18,24,9,16
21	19045662	.	A	G	.	TAC;FRQ;1PS	SOMATIC;SNP;AF=0.00,0.02;MQ=60	GT:DP:DP4	0/0:58:32,26,0,0	0/0:109:60,47,1,1
21	19046286	.	A	T	.	PASS	SOMATIC;SNP;AF=0.00,0.14;MQ=60	GT:DP:DP4	0/0:37:23,14,0,0	0/1:64:39,16,8,1
21	19048103	.	T	G	.	RE;TAC;FRQ;1PS	SOMATIC;SNP;AF=0.00,0.03;MQ=59	GT:DP:DP4	0/0:44:19,25,0,0	0/0:69:38,29,1,1
21	19050714	.	G	T	.	RE;TAC;SBAF;FRQ	SOMATIC;SNP;AF=0.00,0.03;MQ=60	GT:DP:DP4	0/0:35:10,25,0,0	0/0:86:18,65,0,3
21	19070737	.	A	C	.	RE;TAC;SBAF;FRQ	SOMATIC;SNP;AF=0.00,0.04;MQ=60	GT:DP:DP4	0/0:30:20,10,0,0	0/0:72:47,22,0,3
21	19072796	.	T	G	.	RE;TAC;SBAF;FRQ	SOMATIC;SNP;AF=0.00,0.02;MQ=60	GT:DP:DP4	0/0:47:20,27,0,0	0/0:109:48,59,2,0
21	19082137	.	G	T	.	RE;TAC	SOMATIC;SNP;AF=0.00,0.05;MQ=60	GT:DP:DP4	0/0:51:21,30,0,0	0/0:107:55,47,3,2
21	19082530	.	C	T	.	RE;TAC;SBAF;FRQ	SOMATIC;SNP;AF=0.00,0.03;MQ=60	GT:DP:DP4	0/0:46:18,28,0,0	0/0:86:37,46,3,0
21	19100393	.	T	G	.	RE;TAC;FRQ;1PS	SOMATIC;SNP;AF=0.00,0.02;MQ=60	GT:DP:DP4	0/0:54:38,16,0,0	0/0:106:75,29,1,1
21	19117363	.	A	C	.	RE;TAC;FRQ;1PS;BI	SOMATIC;SNP;AF=0.00,0.02;MQ=60	GT:DP:DP4	0/0:37:21,16,0,0	0/0:90:54,34,1,1
21	19118173	.	G	A	.	RE;TAC;FRQ;1PS	SOMATIC;SNP;AF=0.00,0.02;MQ=60	GT:DP:DP4	0/0:53:31,22,0,0	0/0:114:44,68,1,1
21	19121278	.	G	T	.	RE;TAC;MAP;SBAF;FRQ	SOMATIC;SNP;AF=0.00,0.02;MQ=49	GT:DP:DP4	0/0:68:31,37,0,0	0/0:97:57,38,2,0
21	19134437	rs75915658_19134437	A	C	.	RE;VAF	SOMATIC;SNP;AF=0.02,0.06;MQ=60;DB	GT:DP:DP4	0/0:51:21,29,0,1	0/0:103:40,57,1,5
21	19169178	.	C	A	.	TAC;FRQ	SOMATIC;SNP;AF=0.00,0.02;MQ=60	GT:DP:DP4	0/0:68:28,40,0,0	0/0:139:62,74,1,2
21	19173632	.	G	T	.	RE;TAC;MAP;SBAF;FRQ;BI	SOMATIC;SNP;AF=0.00,0.02;MQ=60	GT:DP:DP4	0/0:37:21,16,0,0	0/0:94:49,43,2,0
21	19179703	.	C	A	.	RE;TAC;FRQ	SOMATIC;SNP;AF=0.00,0.04;MQ=60	GT:DP:DP4	0/0:56:30,26,0,0	0/0:115:53,57,3,2
21	19186534	.	A	C	.	TAC;SBAF;FRQ;BI	SOMATIC;SNP;AF=0.00,0.03;MQ=60	GT:DP:DP4	0/0:43:23,20,0,0	0/0:95:52,40,0,3
21	19186574	.	G	A	.	TAC;SBAF;FRQ	SOMATIC;SNP;AF=0.00,0.02;MQ=60	GT:DP:DP4	0/0:48:17,31,0,0	0/0:117:44,71,0,2
21	19197993	.	T	G	.	RE;TAC;SBAF;FRQ;VAF	SOMATIC;SNP;AF=0.02,0.03;MQ=60	GT:DP:DP4	0/0:64:27,36,1,0	0/0:148:46,98,4,0
21	19207689	.	C	G	.	PASS	SOMATIC;SNP;AF=0.00,0.09;MQ=60	GT:DP:DP4	0/0:62:30,32,0,0	0/0:122:56,55,3,8
21	19221500	.	T	A	.	RE;TAC;SBAF;FRQ	SOMATIC;SNP;AF=0.00,0.02;MQ=60	GT:DP:DP4	0/0:19:13,6,0,0	0/0:83:72,9,2,0
21	19273105	.	T	G	.	RE;TAC;SBAF;FRQ;BI	SOMATIC;SNP;AF=0.00,0.02;MQ=60	GT:DP:DP4	0/0:36:20,16,0,0	0/0:97:49,46,2,0
21	19283214	.	C	A	.	TAC;SBAF;FRQ	SOMATIC;SNP;AF=0.00,0.02;MQ=60	GT:DP:DP4	0/0:50:22,28,0,0	0/0:118:47,69,2,0
21	19312319	.	T	G	.	RE;TAC;SBAF;FRQ	SOMATIC;SNP;AF=0.00,0.02;MQ=60	GT:DP:DP4	0/0:55:25,30,0,0	0/0:94:27,65,2,0
21	19316282	.	C	T	.	PASS	SOMATIC;SNP;AF=0.00,0.38;MQ=60	GT:DP:DP4	0/0:87:43,44,0,0	0/1:159:53,45,27,34
21	19335010	.	A	T	.	TAC;SBAF;FRQ;TAR	SOMATIC;SNP;AF=0.00,0.01;MQ=60	GT:DP:DP4	0/0:62:15,47,0,0	0/0:116:35,80,1,0
21	19351068	.	C	A	.	RE;TAC;SBAF;FRQ	SOMATIC;SNP;AF=0.00,0.03;MQ=60	GT:DP:DP4	0/0:44:23,21,0,0	0/0:102:54,45,3,0
21	19360873	.	A	C	.	RE;SB;TAC;MAP;FRQ	SOMATIC;SNP;AF=0.00,0.33;MQ=40	GT:DP:DP4	0/0:6:6,0,0,0	0/1:9:6,0,3,0
21	19414361	.	C	G	.	PASS	SOMATIC;SNP;AF=0.00,0.19;MQ=60	GT:DP:DP4	0/0:30:11,19,0,0	0/1:21:3,14,2,2
21	19449005	.	T	A	.	RE;TAC;SBAF;FRQ;TAR;BI	SOMATIC;SNP;AF=0.00,0.02;MQ=60	GT:DP:DP4	0/0:59:34,25,0,0	0/0:51:27,23,0,1
21	19450433	.	A	G	.	RE;TAC;FRQ;1PS	SOMATIC;SNP;AF=0.00,0.03;MQ=60	GT:DP:DP4	0/0:68:33,35,0,0	0/0:71:35,34,1,1
21	19472161	.	G	A	.	PASS	SOMATIC;SNP;AF=0.00,0.26;MQ=60	GT:DP:DP4	0/0:51:23,28,0,0	0/1:38:13,15,6,4
21	19499226	.	A	C	.	PASS	SOMATIC;SNP;AF=0.00,0.23;MQ=60	GT:DP:DP4	0/0:51:37,14,0,0	0/1:35:19,8,6,2
21	19544778	.	A	T	.	PASS	SOMATIC;SNP;AF=0.00,0.33;MQ=60	GT:DP:DP4	0/0:51:28,23,0,0	0/1:49:14,19,8,8
21	19866637	.	G	T	.	PASS	SOMATIC;SNP;AF=0.00,0.38;MQ=58	GT:DP:DP4	0/0:28:14,14,0,0	0/1:29:13,5,8,3
21	19873456	.	A	G	.	PASS	SOMATIC;SNP;AF=0.00,0.23;MQ=60	GT:DP:DP4	0/0:69:34,35,0,0	0/1:30:11,12,4,3
21	19885205	rs374936781_19885205	C	A	.	RE;TAC;SBAF;FRQ	SOMATIC;SNP;AF=0.00,0.09;MQ=60;DB	GT:DP:DP4	0/0:48:24,24,0,0	0/0:35:13,19,3,0
```
According to the result, we found that `HCC1143.tumor.21.19M-20M.NeuSomatic.vcf` 100% match `Control.vcf`  base on "PASS" status 

### Perform an analysis on synthetic data

We use [Vcftools](https://vcftools.github.io/man_latest.html#EXAMPLES) to compare 

Command to test 

` vcftools --vcf BSdata/truth.testBS.vcf  --diff  BSdata/work_call/tumor.sorted.testBS.NeuSomatic.vcf --diff-site `

```
# From out.diff.sites_in_files
Comparing sites in VCF files...
Found 99 sites common to both files.
Found 2 sites only in main file.
Found 0 sites only in second file.
Found 0 non-matching overlapping sites.
After filtering, kept 101 out of a possible 101 Sites
Run Time = 0.00 seconds
...
22	33949802	33949802	B	T	T	G	G
22	33956547	33956547	B	G	G	T	T
22	33958087	.	1	T	.	A	.
22	33960216	33960216	B	T	T	G	G
22	33964202	33964202	B	G	G	T	T
22	33965973	33965973	B	G	G	T	T
22	33971956	33971956	B	T	T	C	C
22	33976718	33976718	B	A	A	G	G
22	33994140	33994140	B	A	A	T	T
22	33994168	33994168	B	C	C	G	G
22	34006643	34006643	B	A	A	T	T
22	34022812	34022812	B	T	T	G	G
22	34029109	34029109	B	T	T	C	C
22	34044952	34044952	B	A	A	G	G
22	34056454	34056454	B	G	G	C	C
22	34065260	34065260	B	A	A	G	G
22	34067333	34067333	B	A	A	T	T
22	34067772	34067772	B	A	A	T	T
22	34087115	34087115	B	T	T	G	G
22	34100303	34100303	B	C	C	G	G
22	34113506	34113506	B	T	T	G	G
22	34139483	34139483	B	C	C	T	T
22	34141184	34141184	B	G	G	C	C
22	34148802	34148802	B	A	A	G	G
22	34166720	.	1	T	.	A	.
...
```

According to the result, we found two positons weren't called by NeuSomatic
`33958087 and 34166720  `


## NeuSomatic - Troubleshooting
* More error message can be inspected at scan.err file (e.g work_call\work_normal\work.1)
* Some error can be solved by deleting whole work_call directory after that rerun again

## BamSurgeon - Troubleshooting
**None**

## Changelog
* 10-5-2020 first submit


<span style="font-family:Papyrus; font-size:4em;">LOVE :heart:  :tw: :kr: :th: :cn: !</span>
