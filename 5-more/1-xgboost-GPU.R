
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



## monitor with:      while true; do gpustat| grep Tesla; sleep 1; done
##                    mpstat 1

n_trees <- 1000    ## use larger number of trees (even if overfitting) if you want to see/study CPU/GPU utilization and relative runtime

## you can also see GPU/CPU relative runtime for larger training sets (going in favor for GPUs)
## change    d_train <- fread("https://s3.amazonaws.com/benchm-ml--main/train-0.1m.csv")   to  ...train-1m.csv...   above
## and also   n_trees <- 100   in this case if you don't want to wait for long running times



## CPU
system.time({
  md <- xgb.train(data = dxgb_train, objective = "binary:logistic", 
           nround = n_trees, max_depth = 10, eta = 0.1,
           tree_method = "hist")
})
phat <- predict(md, newdata = X_test)
rocr_pred <- prediction(phat, d_test$dep_delayed_15min)
performance(rocr_pred, "auc")@y.values[[1]]



## GPU
system.time({
  md <- xgb.train(data = dxgb_train, objective = "binary:logistic", 
           nround = n_trees, max_depth = 10, eta = 0.1,
           tree_method = "gpu_hist")
})
phat <- predict(md, newdata = X_test)
rocr_pred <- prediction(phat, d_test$dep_delayed_15min)
performance(rocr_pred, "auc")@y.values[[1]]



