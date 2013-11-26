library(data.table)
rawTrain <- data.table(read.csv("data/train.csv", stringsAsFactors = FALSE))
rawTrain$state <- as.factor(rawTrain$state)

rawTest <- data.table(read.csv("data/test.csv", stringsAsFactors = FALSE))
rawTest$state <- as.factor(rawTest$state)

save(rawTrain, rawTest, file="data/rawData.RData")
