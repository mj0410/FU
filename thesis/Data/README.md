## Databases

### Synthea - synthetic patients data generator
[Synthea paper](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7651916/) </br> 
[Synthea data](https://synthea.mitre.org/) </br>
[Synthea csv data dictionary](https://github.com/synthetichealth/synthea/wiki/CSV-File-Data-Dictionary) </br>
[How to generate data](https://github.com/synthetichealth/synthea/wiki/Basic-Setup-and-Running)
```
java -jar synthea-with-dependencies.jar -p NUMBER_OF_PATIENTS --exporter.csv.export true
```

### MIMIC database
[MIMIC-III](https://mimic.mit.edu/docs/)</br>
<sup> weird date of birth / death of patients [Explanation](https://github.com/MIT-LCP/mimic-code/issues/637) </sup>


## Synthetic data generators
> Generate synthetic data using given data
> We may give MIMIC demo files and get synthetic one

### Gretel - synthetic data generation
[Gretel](https://synthetics.docs.gretel.ai/en/stable/index.html)

### synthetic_data
[paper](https://hal.inria.fr/hal-03158556/file/Synthesizing_ICBIS_2020.pdf)
[github](https://github.com/TheRensselaerIDEA/synthetic_data)

