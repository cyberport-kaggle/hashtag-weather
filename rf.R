library(randomForest)
library(doMC)
registerDoMC(4)

trainX <- read.table("data/lsa_train_x.csv", header=FALSE, sep=',', nrows=80000, colClasses='numeric')
trainY <- read.table("data/lsa_train_y.csv", header=FALSE, sep=',', nrows=80000, colClasses='numeric')
testX <- read.table("data/lsa_test_x.csv", header=FALSE, sep=',', nrows=50000, colClasses='numeric')

id <- read.csv("data/sampleSubmission.csv", colClasses=c("numeric", rep(NULL, 24)))[,1]

res <- matrix(0, nrow=nrow(testX), ncol=ncol(trainY))

for(i in 1:ncol(trainY))
{
  cat("Training column", i, "\n")
  rfFit <- foreach(ntree = rep(24,4), .combine=combine, .packages='randomForest') %dopar% {
    randomForest(x=trainX, y=trainY[,i], ntree=ntree, do.trace=TRUE)   
  }
  res[,i] <- predict(rfFit, testX)
}

cnames <- c("id",
            paste0(rep("s", 5), 1:5),
            paste0(rep("w", 4), 1:4),
            paste0(rep("k", 15), 1:15))
res <- cbind(id, res)
colnames(res) <- cnames
write.csv(res, "results/rf.csv", col.names = TRUE, row.names = FALSE)
