
library(h2o)

h2o.init()    ## starts h2o "server" (java) and connects to it from R


dx_train <-  h2o.importFile("https://s3.amazonaws.com/benchm-ml--main/train-0.1m.csv")  ## the data is on the server, not in R
dx_test <-  h2o.importFile("https://s3.amazonaws.com/benchm-ml--main/test.csv") 


## no need e.g. for 1-hot encoding, h2o deals with it :)


## TRAIN
system.time({
  md <- h2o.gbm(x = 1:(ncol(dx_train)-1), y = "dep_delayed_15min", training_frame = dx_train, distribution = "bernoulli", 
                ntrees = 100, max_depth = 10, learn_rate = 0.1, 
                nbins = 100, seed = 123)    
})
md


## SCORE
##phat <- h2o.predict(md, dx_test)

h2o.auc(h2o.performance(md, dx_test))


## inspect in web UI (Flow):  http://localhost:54321

## can export model in POJO/MOJO for fast scoring from Java
## with "steam" one can build a real-time scoring web service (REST API)




