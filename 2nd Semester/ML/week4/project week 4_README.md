# Week4 - [Video Report ]()

>[![ForTheBadge built-by-developers](http://ForTheBadge.com/images/badges/built-by-developers.svg)](https://GitHub.com/Naereen/)  

🤖 TeamE 🤖 

----

## Some questions you should be able to answer

1. What is Mass Spectrometry (MS) and how is it used in proteomics research?
>Mass spectrometry is a sensitive technique used to detect, identify and quantitate molecules based on their mass-to-charge (m/z) ratio. Proteomics is the study of all proteins in a biological system (e.g., cells, tissue, organism) during specific biological events. MS was used to sequence oligonucleotides and peptides and analyze nucleotide structure. Ionization also allowed scientists to obtain protein mass "fingerprints" that could be matched to proteins and peptides in databases and help identity unknown targets. New isotopic tagging methods led to the quantitation of target proteins both in relative and absolute quantities.
 ```
  Common applications and fields that use mass spectrometry
 1. Determine protein structure, function, folding and interactions.
 2. Identify a protein from the mass of its peptide fragments.
 3. Detect specific post-translational modifications throughout complex biological mixtures using      Workflows for phosphoproteomics and protein glycosylation.
 4. Quantitate proteins (relative or absolute) in a given sample.
 5. Monitor enzyme reactions, chemical modifications and protein digestion.
```
2. How does a typical MS data-set look like?

The output type of a mass spectrometer varies depending on the instrument vendor, and the first step of most workflows therefore consists of converting these raw (binary) files into a standard (binary) files into a standard open format,often **mzML** An mzML file contains all the unprocessed spectra (MS1 and MS2) plus additional spectrum and instrument annotation.
However, The community often prefers the simpler **mgf** format for spectrum identification.

The mgf file contains all MS/MS spectra in the dataset. Each spectrum starts with the line "BEGIN IONS", followed by 5 header lines:

* "TITLE": not relevant
* "PEPMASS": the center of DIA m/z window of the spectrum
* "CHARGE": not relevant
* "SCANS": the MS/MS scan id. For example, "F1:3" means scan number 3 of fraction 1.
* "RTINSECONDS": the retention time of the spectrum
After those headers lines comes the list of pairs of (m/z, intensity) of fragment ions in the spectrum. Finally, the spectrum ends with the line "END IONS"


```

#### For example;

BEGIN IONS
TITLE=File_A_Spectrum_1
RTINSECONDS=173.824
PEPMASS=1467.61962890625 1671.478515625
CHARGE=3
438.2539978 3469.398926
470.861908  1319.134888
.
.
1986.876587 2016.473755
END IONS

```
[**Data file format**](http://www.matrixscience.com/help/data_file_help.html)

3. What is the difference between 

1D-MS data (e.g. MALDI-MS), 

2D-MS data (e.g. LC-MS) and

MS/MS (orMS^2) data?

>Tandem mass spectrometry, also known as MS/MS or MS2, is a technique in instrumental analysis where two or more mass analyzers are coupled together using an additional reaction step to increase their abilities to analyse chemical samples. A common use of tandem-MS is the analysis of biomolecules, such as proteins and peptides.



4. What is data-independent acquisition (DIA) In mass spectrometry?
>In mass spectrometry, data-independent acquisition (DIA) is a method of molecular structure determination in which all ions within a selected m/z range are fragmented and analyzed. In DIA mode, for each cycle, the instrument focuses on a narrow mass window of precursors and acquires MS/MS data from all precursors detected within that window. This mass window is then stepped across the entire mass range, systematically collecting MS/MS data from every mass (m/z) and from all detected precursors.

5. What is the difference between DIA and DDA?
>Traditional data-dependent acquisition (DDA) takes only a selection of peptide signals forward for fragmentation, and then matches them to a pre-defined database.  In contrast, DIA fragments every single peptide in a sample. It therefore unbiased, in theory making it the better technique for discovery proteomics. 

6. What is the input to the DeepNovo-DIA algorithm?

The precursor (.csv)and its associated MS/MS spectra (.mgf)


7. What is the output of the DeepNovo-DIA algorithm?

The result is a tab-delimited text file with extension .deepnovo_denovo. Each row includes the following columns:

```

feature_id
feature_area
predicted_sequence
predicted_score
predicted_position_score: positional score for each amino acid in predicted_sequence
precursor_mz
precursor_charge
protein_access_id: not relevant
scan_list_original: list of scan ids of DIA spectra associated with this feature
scan_list_middle: list of DIA spectra used for de novo sequencing
predicted_score_max: same as predicted_score, not relevant
```

list of precursor features, each of which should include the following information: feature ID, m/z, charge, abundance level (area), retention-time center, and intensity values over the retention-time range

```
“spec_group,” the feature ID; “F1:6427” means feature number 6,427 of fraction 1

m/z, the mass-to-charge ratio

z, the charge

“rt_mean,” the mean of the retention-time range

“seq”: the column is empty during de novo sequencing. In training mode, it contains the peptides identified by the in-house database search for training.

“scans,” a list of all MS/MS spectra collected for the feature as described above. The spectra’s IDs are separated by a semicolon; “F1:101” indicates scan number 101 of fraction 1. The spectra’s IDs can be used to locate the spectra in the MGF file “testing_plasma.spectrum.mgf.”

“profile,” the intensity values over the retention-time range; the values are “time:intensity” pairs and are separated by semicolons; the time points align to the time of spectra in the column “scans.”

“feature_area,” the precursor feature area estimated by the feature detection
```

8. How is the confidence score of a peptide sequence and its individual amino acids computed?

>The confidence score of a peptide sequence is the sum of its amino acids’ scores. The score of each amino acid is the log of the output probability distribution—that is, the final softmax layer of the neural network model—at each sequencing iteration. The score was trained using only the training dataset.

9. What precursor and fragment ions?

**precursor ion.** = In mass spectrometry, the ion that dissociates to a smaller fragment ion, usually due to collision-induced dissociation in an multistage/mass spectrometry (MS/MS) experiment.

**fragment ion** = the context of mass spectrometry as the charged product of an ion dissociation. A fragment ion may be stable or dissociate further to form other charged fragment ions and neutral species of successively lower mass.

10. Why is it a good idea to complement genomics based somatic mutation calling with MS
technology?

> MS can  identify peptide ligands that presented on tumour such as neoantigens so together with somatic mutation calling(DNA level) we  can possibly conclude that a mutation has siginificat effect and high relevance for cancer development. This can yield more accuracy of identifiaction.

11. What is the false discovery rate (FDR)?

> The false discovery rate is the ratio of the number of false positive results to the number of total positive test results.

12. What is the connection between an RNN and an LSTM and how is it different to a feedforward network?

> In a feedforward neural network, signals flow in only one direction from input to output, one layer at a time.  In contrast, RNNs have a feedback loop where the net's output is fed back into the net along with the next input. A recurrent net is suited for time series data, where an output can be the next value in a sequence, or the next several values. It can not only process single data points (such as images), but also entire sequences of data (such as speech or video).
LSTM networks are well-suited to classifying, processing and making predictions based on time series data, since there can be lags of unknown duration between important events in a time series. LSTMs were developed to deal with the vanishing gradient problem that can be encountered when training traditional RNNs. Relative insensitivity to gap length is an advantage of LSTM over RNNs, hidden Markov models and other sequence learning methods in numerous applications



**Read more**

1. [Bioinformatic analysis of proteomics data](https://bmcsystbiol.biomedcentral.com/articles/10.1186/1752-0509-8-S2-S3)
2. [proteomics-ms-overview](http://www.mi.fu-berlin.de/wiki/pub/ABI/QuantProtP4/proteomics-ms-overview.pdf)

3. [Biological_MS_and_Proteomics](https://www.broadinstitute.org/files/shared/proteomics/Fundamentals_of_Biological_MS_and_Proteomics_Carr_5_15.pdf)

4. [The New BLAST® Results Page](https://ftp.ncbi.nlm.nih.gov/pub/factsheets/HowTo_BLAST_NewResultPage.pdf)
----


## How to install

see this https://github.com/nh2tran/DeepNovo-DIA

or 

Download from this [Google Drive]( https://drive.google.com/open?id=1T07-YHvJdmSE1emx8U8YmYrtq0Z1mEbN)

**[Achtung!!:warning:]** don't use windows version 


## How to run

```bash

## Run de novo sequencing with a pre-trained model:
./deepnovo_main --search_denovo \
--train_dir train.urine_pain.ioncnn.lstm \
--denovo_spectrum oc/testing_oc.spectrum.mgf \
--denovo_feature oc/testing_oc.feature.csv 

## Test de novo sequencing results on labeled features
./deepnovo_main --test --target_file  oc/testing_oc.feature.csv --predicted_file oc/testing_oc.feature.csv.deepnovo_denovo

```

### To select high confident denovo peptide with spectifc confident 

download https://static-content.springer.com/esm/art%3A10.1038%2Fs41592-018-0260-3/MediaObjects/41592_2018_260_MOESM4_ESM.zip

and use `deepnovo_dia_script_select.py` script to select

## Evaluate result


### Compare performance with original paper 

In the paper, they measured the sequencing accuracy at
the amino acid level (i.e., the ratio of the total number of matched
amino acids to the total length of predicted peptides) and at the peptide
level (i.e., the fraction of fully matched peptides).

According to deepnovo_worker_test.py, we can check 'precision_AA_mass_db' and 'precision_peptide_mass_db' can represent the accuracy.

```bash
#recall_AA_total = total number of matched amino acids
#recall_peptide_total = total number of fully matched peptides
#predicted_len_mass_db = number of predicted amino acids
#predicted_count_mass_db = number of predicted peptides

print("precision_AA_mass_db  = {0:.4f}".format(recall_AA_total / predicted_len_mass_db))
print("precision_peptide_mass_db  = {0:.4f}".format(recall_peptide_total / predicted_count_mass_db))

```
![image](https://im2.ezgif.com/tmp/ezgif-2-e16307b2c80b.png) 
##### Result from oc

![image](https://im2.ezgif.com/tmp/ezgif-2-6d9eac14c876.png) 
##### Result from uti

![image](https://im2.ezgif.com/tmp/ezgif-2-dff2fcf70b93.png) 
##### Result from plasma

![image](https://im2.ezgif.com/tmp/ezgif-2-0948432e0f0e.png)
##### Result from paper

We can check our result is as same as in the paper.


### Using Blast search 

```
1.Protein name
2.Gene associated 
3.Protein databasae where from  e.g. Uniport
4.Sequence ID:  XM number AA number
5.Sequence
Identities -maximize as high as possible
Gaps - minimise as low as possible
Expect (e-value)  as low as possible  (statistically significant)
6.Identifying species
7.Locating domains (protein domain , conserved part )
```


----

<span style="font-family:Papyrus; font-size:4em;">LOVE :heart:  :tw: :kr: :th: :cn: !</span>
