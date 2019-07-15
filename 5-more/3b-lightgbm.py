
import pandas as pd
import numpy as np
from sklearn import preprocessing 
from scipy import sparse
from sklearn import metrics

import lightgbm as lgb


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
md = lgb.LGBMClassifier(num_leaves=512, learning_rate=0.1, n_estimators=100)
%time md.fit(X_train, y_train)

## %time md.fit(X_train.toarray(), y_train)   # slow if not sparse!


## SCORE
y_pred = md.predict_proba(X_test)[:,1]

print(metrics.confusion_matrix(y_test, y_pred>0.7))
print(metrics.roc_auc_score(y_test, y_pred))



## Method 2 - orig lightgbm API

dlgb_train = lgb.Dataset(X_train, label = y_train)
dlgb_test = lgb.Dataset(X_test)


## TRAIN
param = {'num_leaves':512, 'learning_rate':0.1, 'verbose':0}
%time md = lgb.train(param, dlgb_train, num_boost_round = 100)


## SCORE
y_pred = md.predict(X_test)   

metrics.confusion_matrix(y_test, y_pred>0.5)
metrics.roc_auc_score(y_test, y_pred)


