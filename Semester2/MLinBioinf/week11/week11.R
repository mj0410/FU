library(xcms)


savePath <- "C:/Users/minie/Desktop/week11/all_XCMS.csv"
workers <- 4
dataDirectory <- "C:\\Users\\minie\\Desktop\\week11\\original"
setwd(dataDirectory)

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

#  Matches ("groups") peaks across samples (rtCheck = maximum amount of time from the median RT)

# density method
gds<-group(ds, method="density",
           minfrac=0,
           minsamp=0,
           bw=1,
           mzwid=0.01,
           sleep=0
)


#   identify peak groups and integrate samples
fds <- fillPeaks(gds, method="chrom", BPPARAM=SnowParam (workers = workers))

write.csv(peakTable(fds), file="C:/Users/minie/Desktop/all_XCMS.csv")
