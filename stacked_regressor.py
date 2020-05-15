from __future__ import division

from optparse import OptionParser
import pandas as pd
import pickle
import psutil
import json
import os
import gc

import numpy as np

from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import classification_report, \
                            confusion_matrix, \
                            roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline

from utils.stacking import StackingAveragedModels
from utils.metrics import rmse

import lightgbm as lgb
import xgboost as xgb

from config import XGB_PARAMS

parser = OptionParser()

parser.add_option("-p", dest="path", help="Path to the csv.")
(options, args) = parser.parse_args()

if not options.path:
    parser.error("You must pass -p argument")

common_path = os.path.join('logs', 'stacked_regressor')
if not os.path.exists(common_path):
    os.makedirs(common_path)

data = pd.read_csv(options.path)

# by default, select the last column to be the target and other columns to be the source data.
# Note that there is no preprocessing here. You need to implement your own
label_column = data.columns[-1]
x = data.drop(labels=[label_column], axis=1)
y = data.iloc[:, -1]

# by default, check if the label contains numbers. If don't, so automatically label encode it
if y.dtype.name == "object":
    lbl = LabelEncoder()
    lbl.fit(y.values)
    y = lbl.transform(y.values)

    # save to use later
    with open(os.path.join(common_path, 'target_encoder.pickle'), 'wb') as f:
        pickle.dump(lbl, f)
else:
    y = y.values

# check if there is any column that contains categorical features. If yes, so label encode it
# and dump the LabelEncoder to use later
try:
    for column in x.dtypes[data.dtypes == "object"].index:

        # Use LabelEncoder to transform categorical into numerical
        lbl = LabelEncoder()
        lbl.fit(list(x[column].values))
        x[column] = lbl.transform(list(x[column].values))

        # pickle the encoder to use later
        with open(os.path.join(common_path, column + '.pickle'), 'wb') as f:
            pickle.dump(lbl, f)
except:
    # probably x is a series
    pass

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

#Validation function
n_folds = 5

def rmsle_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(x_train)
    rmse = np.sqrt(-cross_val_score(model, x_train, y_train, scoring="neg_mean_squared_error", cv=kf))
    return rmse

# Base models
lasso = make_pipeline(RobustScaler(), Lasso(alpha=0.0005, random_state=1))

ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))

KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)

GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber', random_state=5)

model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=3, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state=7, nthread=-1)

model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin=55, bagging_fraction=0.8,
                              bagging_freq=5, feature_fraction=0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf=6, min_sum_hessian_in_leaf=11)

stacked_averaged_models = StackingAveragedModels(base_models=(ENet, GBoost, KRR), meta_model=lasso)

# training step
print("Traininf stacked model...")
stacked_averaged_models.fit(x_train, y_train)

print("Training xgb model...")
model_xgb.fit(x_train, y_train)

print("Training lgb model..")
model_lgb.fit(x_train, y_train)

# save models
with open(os.path.join(common_path, 'xgb_regressor.pickle'), 'wb') as f:
    pickle.dump(model_xgb,f)
with open(os.path.join(common_path, 'lgb_regressor.pickle'), 'wb') as f:
    pickle.dump(model_lgb,f)

stacked_averaged_models.save(common_path, ["enet, gboost, krr"], "lasso")

# eval step
print("Predicting...")
stacked_train_pred = stacked_averaged_models.predict(x_test)
xgb_train_pred = model_xgb.predict(x_test)
lgb_train_pred = model_lgb.predict(x_test)

print("Stacked RMSE:", rmse(y_test, stacked_train_pred))
print("XGBRegressor RMSE:", rmse(y_test, xgb_train_pred))
print("LGBRegressor RMSE:", rmse(y_test, lgb_train_pred))
print("Averaged score:", rmse(y_test, stacked_train_pred*0.65 + xgb_train_pred+0.2 + lgb_train_pred*0.15))

result = pd.DataFrame(
    data=[
        rmse(y_test, stacked_train_pred),
        rmse(y_test, xgb_train_pred),
        rmse(y_test, lgb_train_pred),
        rmse(y_test, stacked_train_pred*0.65 + xgb_train_pred+0.2 + lgb_train_pred*0.15)
    ],
    columns=[
        'stacked-rsme',
        'xgbregressor-rmse',
        'lgbregressor-rmse',
        'averaged-rmse'
    ]
)

result.to_csv(os.path.join(common_path, 'result.csv'), index=False, encoding='utf-8')