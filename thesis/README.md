## Development and evaluation of a machine learning toolbox for medical data integration

### Goal
> Automation of tasks to integrate digital health data </br>
> Create an integrated table of good quality with the application of machine learning

### Architecture

<img src="https://user-images.githubusercontent.com/66175878/145218356-c1bf2f93-651f-4f4c-8269-3ff3faf4072f.png" width=80% height=80%>

````diff
- Input
tables in csv format

- DeepTable
Classify table orientation (horizontal or vertical or matrix)
Transform table to horizontal if it isn’t

- Sato
Detect semantic type of columns
Types : date of birth, date of death, patient id, medication, clinical notes, etc.

- Integration
Integrate tables based on predicted semantic types
(We could add one more ML tools for better outcome)

- Output
A single integrated table
````


### Interesting tools

***Sato: Contextual Semantic Type Detection in Tables (2020)*** [link](http://www.vldb.org/pvldb/vol13/p1835-zhang.pdf)
> <sup> [Sato github](https://github.com/megagonlabs/sato) </br>
> [Sherlock github](https://github.com/mitmedialab/sherlock-project) </sup>

***DeepTable: a permutation invariant neural network for table orientation classification (2020)*** [link](https://link.springer.com/content/pdf/10.1007/s10618-020-00711-x.pdf)
> <sup> [DeepTable github](https://github.com/Marhabibi/DeepTable) </sup>

***SMAT: An attention-based deep learning solution to the automation of schema matching (2021)*** [link](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8487677/)
> <sup> [SMAT github](https://github.com/JZCS2018/SMAT) </sup>

***Automating the Transformation of Free-Text Clinical Problems into SNOMED CT Expressions (2020)*** [link](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7233039/) </br>
***Knowledge Transfer for Entity Resolution with Siamese Neural Networks (2021)*** [link](https://dl.acm.org/doi/pdf/10.1145/3410157?casa_token=niSzKttV1kYAAAAA:5OVRRVQ1pICG_ESHuE93zp_mYjFglNfdchzfryimyJb_rczRQTZXPORgU3yrCIyVbmDHGQih3XZQZg)


***Rotom: A Meta-Learned Data Augmentation Framework for Entity Matching, Data Cleaning, Text Classification, and Beyond (2021)*** [link](https://dl.acm.org/doi/pdf/10.1145/3448016.3457258?casa_token=MpLEGQzhGAUAAAAA:HX_ZqP1e1kZKGgcc26JwUfOXaw8Ir_KKzUrcChdNu99zZFqPekR7XIp8LrDKlaLe1PAT5Fq1trM9eQ)
> <sup> [Rotom github](https://github.com/megagonlabs/rotom) </sup>

***DTranNER: biomedical named entity recognition with deep learning-based label-label transition model (2020)*** [link](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-020-3393-1)
> <sup> [DTranNER github](https://github.com/kaist-dmlab/BioNER) </sup>

### Databases

synthetic EHR [Synthea paper](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7651916/) [Synthea data](https://synthea.mitre.org/)</br>
MIMIC database [MIMIC-III](https://mimic.mit.edu/docs/)</br>
