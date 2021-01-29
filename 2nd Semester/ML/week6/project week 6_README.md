# Week6 - [Video Report ](https://voicethread.com/share/14614475/)

>[![ForTheBadge built-by-developers](http://ForTheBadge.com/images/badges/built-by-developers.svg)](https://GitHub.com/Naereen/)  🤖 TeamE 🤖


----

## Some questions you should be able to answer

1. What is precision oncology? 
> diverse strategies in cancer medicine ranging from the use of targeted therapies
>
> create a treatment plan according to the “precise” molecular aspects of each patient’s cancer

2. How is precision oncology different from classical cancer research?
> use data from next-generation sequencing to select therapy for a person independent of cancer type

3. What is the difference between somatic and germline mutations?
> Somatic mutations – occur in a single body cell and cannot be inherited (only tissues derived from mutated cell are affected)
>
> Germline mutations – occur in gametes and can be passed onto offspring (every cell in the entire organism will be affected)

4. How does a variant caller algorithm work?
>
>
>

5. What is one hot encoding?
> Some algorithms can work with categorical data directly. One hot encoding is a process by which categorical variables are converted into a form that could be provided to ML algorithms to do a better job in prediction.
> Ex) ambiguous(a) = 0, fail(f) = 1, somatic(s) = 2 -> a = (1, 0, 0), f = (0, 1, 0), s = (0, 0, 1)

6. What is a Kappa statistic?
> Kappa statistic (κ) is a measure that takes an expected figure into account by deducting it from the predictor's successes. 
>
> This statistic should only be calculated when:
> * Two raters each rate one trial on each sample.
> * One rater rates two trials on each sample.
>
> The Kappa statistic varies from 0 to 1, where.
> * 0 = agreement equivalent to chance.
> * 0.1 – 0.20 = slight agreement.
> * 0.21 – 0.40 = fair agreement.
> * 0.41 – 0.60 = moderate agreement.
> * 0.61 – 0.80 = substantial agreement.
> * 0.81 – 0.99 = near perfect agreement
> * 1 = perfect agreement.

7. How does the Random Forest method work?
>
>
>

8. What are clinically significant somatic variants?
>
>
>

9. What is the Integrative Geomics Viewer (IGV)?
> The Integrative Genomics Viewer (IGV) is a light-weight, high-performance visualization tool that enables intuitive real-time exploration of diverse, large-scale genomic data sets on standard desktop computers.
> It supports flexible integration of a wide variety of data types including aligned sequence reads, mutations, copy number, RNA interference screens, gene expression, methylation, and genomic annotations.
>
> [IGV](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3346182/)

10. How is the IGV used to manually review potential somatic mutations?
> 
>
>



## How To Run

1. Download all files from [DeepSVR](https://github.com/griffithlab/DeepSVR)
2. Open it through google colab or jupyter notebook
3. Run classifiers 

   **[Achtung!!:warning:]** run without %aimport deepsvr
   
   **[Achtung!!:warning:]** set directories correctly for imported files (.py, pre-trained data)
   
4. Save .npy(predicted) and .pkl(important features) files to compare results
5. Compare results from each classifier


<span style="font-family:Papyrus; font-size:4em;">LOVE :heart:  :tw: :kr: :th: :cn: !</span>
