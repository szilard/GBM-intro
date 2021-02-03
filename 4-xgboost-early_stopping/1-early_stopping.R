
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


## split for early stopping hold-out set
p_subtrain <- 0.8    
idx_subtrain   <- sample(1:nrow(d_train), nrow(d_train)*p_subtrain)
idx_earlystop  <- setdiff(1:nrow(d_train),idx_subtrain)
d_subtrain <- d_train[idx_subtrain,]
d_earlystop <- d_train[idx_earlystop,]
X_subtrain <- X_train[idx_subtrain,]
X_earlystop <- X_train[idx_earlystop,]
dxgb_subtrain  <- xgb.DMatrix(data = X_subtrain,  label = ifelse(d_subtrain$dep_delayed_15min=="Y",1,0))
dxgb_earlystop <- xgb.DMatrix(data = X_earlystop, label = ifelse(d_earlystop$dep_delayed_15min=="Y",1,0))


## TRAIN
system.time({
  md <- xgb.train(data = dxgb_subtrain, objective = "binary:logistic", 
           max_depth = 10, eta = 0.1,
           tree_method = "hist",
           nround = 10000, early_stopping_rounds = 10, watchlist = list(valid = dxgb_earlystop), eval_metric = "auc",  
           verbose = 0)
})
md

md$best_iter


## SCORE
phat <- predict(md, newdata = X_test)
rocr_pred <- prediction(phat, d_test$dep_delayed_15min)
performance(rocr_pred, "auc")@y.values[[1]]


## overfitting

dxgb_test  <- xgb.DMatrix(data = X_test,  label = ifelse(d_test$dep_delayed_15min=="Y",1,0))

system.time({
  md <- xgb.train(data = dxgb_subtrain, objective = "binary:logistic", 
           max_depth = 10, eta = 0.1,
           tree_method = "hist",
           nround = 1000, watchlist = list(valid = dxgb_test), eval_metric = "auc", 
           verbose = 0)
})

ggplot(md$evaluation_log) + geom_line(aes(x = iter, y = valid_auc))





