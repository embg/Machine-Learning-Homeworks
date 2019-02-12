print('laplace = 0')
##################################################
### read in date, make y=ham/spam a factor
smsRaw = read.csv("http://www.rob-mcculloch.org/data/sms_spam.csv", stringsAsFactors = FALSE)
# convert spam/ham to factor.
smsRaw$type = factor(smsRaw$type)


##################################################
### get and clean corpus
# build a corpus using the text mining (tm) package
library(tm)
library(SnowballC)
#volatile (in memory corpus from vector of text in R
smsC = VCorpus(VectorSource(smsRaw$text))
# clean up the corpus using tm_map()
smsCC = tm_map(smsC, content_transformer(tolower)) #upper -> lower
smsCC = tm_map(smsCC, removeNumbers) # remove numbers
smsCC = tm_map(smsCC, removeWords, stopwords()) # remove stop words
smsCC = tm_map(smsCC, removePunctuation) # remove punctuation
smsCC = tm_map(smsCC, stemDocument) #stemming
smsCC = tm_map(smsCC, stripWhitespace) # eliminate unneeded whitespace

##################################################
### create Document Term Matrix
smsDtm = DocumentTermMatrix(smsCC)
dim(smsDtm)
##################################################
### tuning parameter choices
wfreqv = c(5,10,50)
nfr=length(wfreqv)

##################################################
### train/test loop
#convert counts to if(count>0) (yes,no)
convertCounts <- function(x) {
  x <- ifelse(x > 0, "Yes", "No")
}

nsamp = 4 #number of random train/test splits
print('num splits: ');
print(nsamp);

trainfrac = .75 #percent of data to put in train

resM = matrix(0.0,nsamp,nfr) #store out of sample missclassifcation rates here

set.seed(99)
n = nrow(smsDtm) #total sample size
print('numrow (total sample size): ');
print(n);

library(e1071)

# Start the clock!
ptm <- proc.time()
for(i in 1:nsamp) {
   if( (i%%1)==0) cat("on sample ",i,"\n")

   ii = sample(1:n,floor(trainfrac*n))
   smsTrain = smsDtm[ii, ]
   smsTest  = smsDtm[-ii, ]
   smsTrainy = smsRaw[ii, ]$type
   smsTesty  = smsRaw[-ii, ]$type

   for(j in 1:nfr) {
      #pull off columns with frequent words and then convert count to binary
      smsFreqWords = findFreqTerms(smsTrain, wfreqv[j])
      smsFreqTrain = smsTrain[ , smsFreqWords]
      smsFreqTest = smsTest[ , smsFreqWords]
      smsTrainB = apply(smsFreqTrain, MARGIN = 2, convertCounts)
      smsTestB  = apply(smsFreqTest, MARGIN = 2, convertCounts)
      # fit NM on train
      smsNB = naiveBayes(smsTrainB, smsTrainy, laplace = 0)
      # predict on test
      yhat = predict(smsNB,smsTestB)
      # store oos missclass
      ctab = table(yhat,smsTesty)
      misclass = (sum(ctab)-sum(diag(ctab)))/sum(ctab)
      print('misclass for nonparallel:')
      perspam = ctab[2,2]/sum(ctab[,2])
      cat("misclass,perspam: ", misclass,perspam,"\n")
      resM[i,j]=misclass
   }
}
# Stop the clock
tottime = proc.time() - ptm
cat("time:\n")
print(tottime)

#graph results
boxplot(resM)


##################################################
### use for each to run the loop in parallel
library(doParallel)
sessionInfo() #see what packages were loaded
registerDoParallel(cores=4)
cat("number of workers is: ",getDoParWorkers(),"\n")

set.seed(99)
# Start the clock!
ptm <- proc.time()
resP = foreach(i=1:nsamp,.combine=rbind) %do% {
   if( (i%%1)==0) cat("on sample ",i,"\n")

   ii = sample(1:n,floor(trainfrac*n))
   smsTrain = smsDtm[ii, ]
   smsTest  = smsDtm[-ii, ]
   smsTrainy = smsRaw[ii, ]$type
   smsTesty  = smsRaw[-ii, ]$type

   resv=rep(0,nfr)
   for(j in 1:nfr) {
      #pull off columns with frequent words and then convert count to binary
      smsFreqWords = findFreqTerms(smsTrain, wfreqv[j])
      smsFreqTrain = smsTrain[ , smsFreqWords]
      smsFreqTest = smsTest[ , smsFreqWords]
      smsTrainB = apply(smsFreqTrain, MARGIN = 2, convertCounts)
      smsTestB  = apply(smsFreqTest, MARGIN = 2, convertCounts)
      # fit NM on train
      smsNB = naiveBayes(smsTrainB, smsTrainy, laplace = 0)
      # predict on test
      yhat = predict(smsNB,smsTestB)
      # store oos missclass
      ctab = table(yhat,smsTesty)
      misclass = (sum(ctab)-sum(diag(ctab)))/sum(ctab)
      # luis modify #
      print('misclass for parallel:')
      perspam = ctab[2,2]/sum(ctab[,2])
      cat("misclass,perspam: ", misclass,perspam,"\n")
      # end luis modify #
      resv[j]=misclass
   }
   resv
}
# Stop the clock
tottimeP = proc.time() - ptm
cat("time:\n")
print(tottimeP)

## plot results
boxplot(resP)


## compare results
qqplot(as.double(resM),as.double(resP))
abline(0,1)

save(resM,tottime,resP,tottimeP,file="oosloop_cutoff5.RData")


##################################################
### pull off age, adult tables
ii= 1:4169

smsTrain = smsDtm[ii, ]
print(smsTrain[1:3,1:5])

smsTrainy = smsRaw[ii, ]$type

smsFreqWords = findFreqTerms(smsTrain, 5) #words that appear at leat 5 times

smsFreqTrain = smsTrain[ , smsFreqWords]
smsTrainB = apply(smsFreqTrain, MARGIN = 2, convertCounts)

iiy1 = (smsTrainy=="ham")
smsAge1 = smsTrainB[iiy1,"age"]
smsAdult1 = smsTrainB[iiy1,"adult"]
smsAge0 = smsTrainB[!iiy1,"age"]
smsAdult0 = smsTrainB[!iiy1,"adult"]

sink("age-adult-tables5.txt")
table(smsAge1,smsAdult1)
table(smsAge0,smsAdult0)
sink()
