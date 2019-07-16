
import h2o
from h2o.estimators.gbm import H2OGradientBoostingEstimator

h2o.init()    ## starts h2o "server" (java) and connects to it from python


dx_train =  h2o.import_file("https://s3.amazonaws.com/benchm-ml--main/train-0.1m.csv")  ## the data is on the server, not in Python
dx_test =  h2o.import_file("https://s3.amazonaws.com/benchm-ml--main/test.csv") 


## no need e.g. for 1-hot encoding, h2o deals with it :)


## TRAIN
md = H2OGradientBoostingEstimator(distribution = "bernoulli",
    ntrees = 100, learn_rate = 0.1, max_depth = 10,
    nbins = 100, seed = 123)
%time md.train(x = dx_train.col_names[:-1], y = "dep_delayed_15min", training_frame = dx_train)


## SCORE
phatx = md.predict(dx_test)["Y"]

md.model_performance(dx_test).auc()


## inspect in web UI (Flow):  http://localhost:54321

## can export model in POJO/MOJO for fast scoring from Java
## with "steam" one can build a real-time scoring web service (REST API)




