
library(data.table)
library(ggplot2)
library(ROCR)

library(xgboost)


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

dxgb_train <- xgb.DMatrix(data = X_train, label = ifelse(d_train$dep_delayed_15min=='Y',1,0))
## special optimized data structure


## TRAIN
system.time({
  md <- xgb.train(data = dxgb_train, objective = "binary:logistic", 
           nround = 100, max_depth = 10, eta = 0.1)
})
md


## SCORE
phat <- predict(md, newdata = X_test)
summary(phat)


## ANALYZE
table(ifelse(phat>0.5,"pred-Y","pred-N"), d_test$dep_delayed_15min)

rocr_pred <- prediction(phat, d_test$dep_delayed_15min)
plot(performance(rocr_pred, measure = "tpr", x.measure = "fpr"))
performance(rocr_pred, "auc")@y.values[[1]]

table(ifelse(phat>0.7,"pred-Y","pred-N"), d_test$dep_delayed_15min)
table(ifelse(phat>0.6,"pred-Y","pred-N"), d_test$dep_delayed_15min)
table(ifelse(phat>0.5,"pred-Y","pred-N"), d_test$dep_delayed_15min)

ggplot(data.frame(phat=phat, y=d_test$dep_delayed_15min)) + geom_density(aes(x=phat, color=y))



## try playing with the hyperparams e.g. max_depth = 2,5,10,15; eta=0.01,0.03,0.1;
## nround = 100,300,1000; check out further params with ?xgb.train 
## (re-run from "TRAIN" part above)

