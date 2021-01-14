
import pandas as pd
import numpy as np
from sklearn import preprocessing 
from sklearn import metrics

import lightgbm as lgb


## READ DATA
d_train = pd.read_csv("https://s3.amazonaws.com/benchm-ml--main/train-1m.csv")
d_test = pd.read_csv("https://s3.amazonaws.com/benchm-ml--main/test.csv")
d_train


## PRE-PROCESS
d_all = pd.concat([d_train,d_test])

vars_cat = ["Month","DayofMonth","DayOfWeek","UniqueCarrier", "Origin", "Dest"]
vars_num = ["DepTime","Distance"]
for col in vars_cat:
  d_all[col] = preprocessing.LabelEncoder().fit_transform(d_all[col])
  
X_all = d_all[vars_cat+vars_num].to_numpy()      ## numpy 2D array, integer encoding for cats
cat_features = list(range(0,len(vars_cat)))
y_all = np.where(d_all["dep_delayed_15min"]=="Y",1,0)       ## numpy 1D array

X_train = X_all[0:d_train.shape[0],]
y_train = y_all[0:d_train.shape[0]]
X_test = X_all[d_train.shape[0]:(d_train.shape[0]+d_test.shape[0]),]
y_test = y_all[d_train.shape[0]:(d_train.shape[0]+d_test.shape[0])]

X_train


## Method 1 - sklearn API


## TRAIN
md = lgb.LGBMClassifier(num_leaves=512, learning_rate=0.1, n_estimators=100)
%time md.fit(X_train, y_train, categorical_feature=cat_features)


## SCORE
y_pred = md.predict_proba(X_test)[:,1]

print(metrics.roc_auc_score(y_test, y_pred))



## Method 2 - classic lightgbm API ("training API")

dlgb_train = lgb.Dataset(X_train, label = y_train, categorical_feature=cat_features)
dlgb_test = lgb.Dataset(X_test, categorical_feature=cat_features)


## TRAIN
param = {'objective': 'binary', 'num_leaves':512, 'learning_rate':0.1, 'verbose':0}
%time md = lgb.train(param, dlgb_train, num_boost_round = 100, categorical_feature=cat_features)


## SCORE
y_pred = md.predict(X_test)   

metrics.roc_auc_score(y_test, y_pred)


