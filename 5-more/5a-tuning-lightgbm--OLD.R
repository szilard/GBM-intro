
## adapted from https://github.com/szilard/GBM-tune/blob/master/1-train_test-same_yr/run-tuning.R

library(lightgbm)
library(data.table)
library(ROCR)
library(dplyr)


d <- as.data.frame(fread("../data/airline100K.csv"))


set.seed(123)
N <- nrow(d)
idx <- sample(1:N, 0.9*N)
d_train <- d[idx,]
d_test <- d[-idx,]
p <- ncol(d)-1

X <- Matrix::sparse.model.matrix(dep_delayed_15min ~ . - 1, data = d)
X_train <- X[idx,]
X_test <- X[-idx,]

## alternative/"better" encoding (vs 1-hot encoding) using `lgb.prepare_rules`
## see e.g here: https://github.com/szilard/GBM-perf/blob/master/run/3-lightgbm2.R


n_random <- 10

params_grid <- expand.grid(                      # default
  num_leaves = c(100,200,500,1000,2000,5000),    # 31 (was 127)
  learning_rate = c(0.01,0.03,0.1),              # 0.1
  min_data_in_leaf = c(5,10,20,50),              # 20 (was 100)
  feature_fraction = c(0.6,0.8,1),               # 1 
  bagging_fraction = c(0.4,0.6,0.8,1),           # 1
  lambda_l1 = c(0,0,0,0, 0.01, 0.1, 0.3),        # 0
  lambda_l2 = c(0,0,0,0, 0.01, 0.1, 0.3)         # 0
  ## TODO:  
  ## min_sum_hessian_in_leaf
  ## min_gain_to_split
  ## max_bin
  ## min_data_in_bin
)
params_random <- params_grid[sample(1:nrow(params_grid),n_random),]


system.time({
  
  d_res <- data.frame()
  for (krpm in 1:nrow(params_random)) {
    params <- as.list(params_random[krpm,])
    
    ## resample
    n_resample <- 5     ## TODO: 20?
    
    p_subtrain <- 0.8       ## TODO: change? (80-10-10 split now)
    p_earlystop <- 0.1   
    p_modelselec <- 1 - p_subtrain - p_earlystop
    
    mds <- list()
    d_res_rs <- data.frame()
    for (k in 1:n_resample) {
      cat(" krpm:",krpm,"k:",k,"\n")
      
      size <- nrow(d_train)
      idx_subtrain   <- sample(1:size, size*p_subtrain)
      idx_earlystop  <- sample(setdiff(1:size,idx_subtrain), size*p_earlystop)
      idx_modelselec <- setdiff(setdiff(1:size,idx_subtrain),idx_earlystop)
      
      d_subtrain <- d_train[idx_subtrain,]
      d_earlystop <- d_train[idx_earlystop,]
      d_modelselec <- d_train[idx_modelselec,]
      
      X_subtrain <- X_train[idx_subtrain,]
      X_earlystop <- X_train[idx_earlystop,]
      X_modelselec <- X_train[idx_modelselec,]
      
      dlgb_subtrain  <- lgb.Dataset(data = X_subtrain,  label = ifelse(d_subtrain[,p+1]=="Y",1,0))
      dlgb_earlystop <- lgb.Dataset(data = X_earlystop, label = ifelse(d_earlystop[,p+1]=="Y",1,0))
      
      runtm <- system.time({
        md <- lgb.train(data = dlgb_subtrain, objective = "binary",
                        params = params,
                        nrounds = 10000, early_stopping_rounds = 10, valid = list(valid = dlgb_earlystop), 
                        verbose = 0)
      })[[3]]
      
      phat <- predict(md, data = X_modelselec)
      rocr_pred <- prediction(phat, d_modelselec[,p+1])
      auc_rs <- performance(rocr_pred, "auc")@y.values[[1]]
      
      d_res_rs <- rbind(d_res_rs, data.frame(ntrees = md$best_iter, runtm = runtm, auc_rs = auc_rs))
      mds[[k]] <- md
    }  
    d_res_rs_avg <- d_res_rs %>% summarize(ntrees = mean(ntrees), runtm = mean(runtm), auc_rs_avg = mean(auc_rs),
                                           auc_rs_std = sd(auc_rs)/sqrt(n_resample))   # std of the mean!
    
    # consider the model as the average of the models from resamples 
    # TODO?: alternatively could retrain the "final" model on all of data (early stoping or avg number of trees?)
    phat <- matrix(0, nrow = n_resample, ncol = nrow(d_test))
    for (k in 1:n_resample) {
      phat[k,] <- predict(mds[[k]], data = X_test)
    }
    phat_avg <- apply(phat, 2, mean)   
    rocr_pred <- prediction(phat_avg, d_test[,p+1])
    auc_test <- performance(rocr_pred, "auc")@y.values[[1]]
    
    d_res <- rbind(d_res, cbind(krpm, d_res_rs_avg, auc_test))
  }
  
})

d_pm_res <- cbind(params_random, d_res)

d_pm_res %>% arrange(desc(auc_rs_avg))

##fwrite(d_pm_res, file = "res.csv")

## view results from random search (n_random=100) here:
## https://github.com/szilard/GBM-tune/blob/master/1-train_test-same_yr/res.csv
## https://htmlpreview.github.io/?https://github.com/szilard/GBM-tune/blob/master/1-train_test-same_yr/analyze.html
## and more results from a larger experiment here:
## https://github.com/szilard/GBM-tune


