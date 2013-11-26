library(data.table)
library(tm)
library(Matrix)
library(glmnet)

load("data/rawData.RData")

convert_to_sparse <- function(text){
  train <- Corpus(VectorSource(text))
  train <- tm_map(train, tolower)
  train <- tm_map(train, removePunctuation)
  train <- tm_map(train, removeNumbers)
  train <- tm_map(train, removeWords, stopwords("english"))
  train <- tm_map(train, stripWhitespace)
  
  dtm <- DocumentTermMatrix(train, 
                            control = list(bounds = list(global = c(2,Inf)),
                                           weighting = function(x) weightTfIdf(x, normalize = FALSE)))
  dtm.sparse <- sparseMatrix(i=dtm$i, j=dtm$j, x=dtm$v, dims=c(dtm$nrow, dtm$ncol))
  
  return(dtm.sparse)
}

trainx = convert_to_sparse(rawTrain$tweet)
trainy = as.matrix(rawTrain[, -c(1:4), with = FALSE])

ridgefit <- cv.glmnet(trainx, trainy, family="mgaussian", alpha = 0)
