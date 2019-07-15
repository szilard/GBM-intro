
library(data.table)
library(ggplot2)
library(ROCR)

library(lightgbm)


## READ DATA
d_train <- fread("https://s3.amazonaws.com/benchm-ml--main/train-0.1m.csv") 
d_test <- fread("https://s3.amazonaws.com/benchm-ml--main/test.csv") 
head(d_train)


## PRE-PROCESS
d_all <- rbind(d_train,d_test)
X_all <- Matrix::sparse.model.matrix(dep_delayed_15min ~ . - 1, data = d_all)   
## 1-hot encoding with sparse mx (less RAM, faster training vs non-sparse mx) 
## needs to be done *together* (train+test) for alignment (otherwise error for new cats at scoring) - still problem in live scoring scenarios
X_train <- X_all[1:nrow(d_train),]
X_test <- X_all[nrow(d_train)+(1:nrow(d_test)),]
X_train[1:5,1:20]

dlgb_train <- lgb.Dataset(data = X_train, label = ifelse(d_train$dep_delayed_15min=='Y',1,0))
## special optimized data structure


## TRAIN
system.time({
  md <- lgb.train(data = dlgb_train, objective = "binary", 
                  nrounds = 100, num_leaves = 512, learning_rate = 0.1)
})


## SCORE
phat <- predict(md, data = X_test)
summary(phat)


## ANALYZE
table(ifelse(phat>0.5,"pred-Y","pred-N"), d_test$dep_delayed_15min)

rocr_pred <- prediction(phat, d_test$dep_delayed_15min)
plot(performance(rocr_pred, measure = "tpr", x.measure = "fpr"))
performance(rocr_pred, "auc")@y.values[[1]]


