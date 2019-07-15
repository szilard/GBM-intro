
import pandas as pd
import numpy as np
from sklearn import preprocessing 
from scipy import sparse
from sklearn import metrics

import xgboost as xgb


## bug workaround for crash on Mac anaconda, crash not happening in Linux anaconda 
## OMP: Error #15: Initializing libomp.dylib, but found libiomp5.dylib already initialized.
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


## READ DATA
d_train = pd.read_csv("https://s3.amazonaws.com/benchm-ml--main/train-0.1m.csv")
d_test = pd.read_csv("https://s3.amazonaws.com/benchm-ml--main/test.csv")
d_train


## PRE-PROCESS
d_all = pd.concat([d_train,d_test])

vars_cat = ["Month","DayofMonth","DayOfWeek","UniqueCarrier", "Origin", "Dest"]
vars_num = ["DepTime","Distance"]
for col in vars_cat:
  d_all[col] = preprocessing.LabelEncoder().fit_transform(d_all[col])
  
X_all_cat = preprocessing.OneHotEncoder(categories="auto").fit_transform(d_all[vars_cat])   
## 1-hot encoding with sparse mx (less RAM, faster training vs non-sparse mx) 
## needs to be done *together* (train+test) for alignment (otherwise error for new cats at scoring) - still problem in live scoring scenarios
X_all = sparse.hstack((X_all_cat, d_all[vars_num])).tocsr()                               
y_all = np.where(d_all["dep_delayed_15min"]=="Y",1,0)          # numpy array, not mx/DF

X_train = X_all[0:d_train.shape[0],]
y_train = y_all[0:d_train.shape[0]]
X_test = X_all[d_train.shape[0]:(d_train.shape[0]+d_test.shape[0]),]
y_test = y_all[d_train.shape[0]:(d_train.shape[0]+d_test.shape[0])]

X_train[0:10,0:10].todense()


## Method 1 - sklearn API


## TRAIN
md = xgb.XGBClassifier(max_depth=10, n_estimators=100, learning_rate=0.1, n_jobs=-1)
%time md.fit(X_train, y_train)

## %time md.fit(X_train.toarray(), y_train)   # slow if not sparse!


## SCORE
y_pred = md.predict_proba(X_test)[:,1]

print(metrics.confusion_matrix(y_test, y_pred>0.7))
print(metrics.roc_auc_score(y_test, y_pred))



## Method 2 - orig xgboost API

dxgb_train = xgb.DMatrix(X_train, label = y_train)
dxgb_test = xgb.DMatrix(X_test)


## TRAIN
param = {'max_depth':10, 'eta':0.1, 'objective':'binary:logistic', 
             'silent':1}             
%time md = xgb.train(param, dxgb_train, num_boost_round = 100)


## SCORE
y_pred = md.predict(dxgb_test)   

print(metrics.confusion_matrix(y_test, y_pred>0.7))
print(metrics.roc_auc_score(y_test, y_pred))



## try playing with the hyperparams e.g. max_depth = 2,5,10,15; learning_rate=0.01,0.03,0.1;
## n_estimators = 100,300,1000; check out further params in the docs
## (re-run from "TRAIN" part above)


