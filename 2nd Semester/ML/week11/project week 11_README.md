# Week11 - [Video Report]()

>[![ForTheBadge built-by-developers](http://ForTheBadge.com/images/badges/built-by-developers.svg)](https://GitHub.com/Naereen/)  🤖 TeamE 🤖

## How To Run

1. Download raw files from [MetaboLights](https://www.ebi.ac.uk/metabolights/MTBLS1129/files)
2. Extract features with the R package 'XCMS'
```bash
  dataDirectory <- "/path/to/raw/files"
  savePath <- "/path/to/save/output/all_XCMS.csv"

  workers <- 8

  setwd(dataDirectory)

  library(xcms)

  ds <- xcmsSet(method="centWave",
          noise=600,
          ppm=25,
          prefilter=c(8, 6000),
          snthresh = 10,
          peakwidth=c(1.5,5),
          mzdiff=0.01,
          mzCenterFun="wMean",
          integrate=2,
          lock=F,
          fitgauss=F,
      	  BPPARAM=SnowParam (workers = workers),  # number of core processors
          )

  gds<-group(ds, method="density",
          minfrac=0,
          minsamp=0,
          bw=1,
          mzwid=0.01,
          sleep=0
          )

  fds <- fillPeaks(gds, method="chrom", BPPARAM=SnowParam (workers = workers))

  write.csv(peakTable(fds), file=savePath)
```

3. Preprocessing and Quality Control of LC-MS Data
> follow [Preprocessing and Quality Control of LC-MS Data with the nPYc-Toolbox.ipynb](https://github.com/phenomecentre/nPYc-toolbox-tutorials/blob/master/Preprocessing%20and%20Quality%20Control%20of%20LC-MS%20Data%20with%20the%20nPYc-Toolbox.ipynb)
4. run ML_week11_classification.ipynb with output csv file from 3.

<span style="font-family:Papyrus; font-size:4em;">LOVE :heart:  :tw: :kr: :th: :cn: !</span>
