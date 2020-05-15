from __future__ import division

from optparse import OptionParser
import pandas as pd
import pickle
import psutil
import json
import os
import gc

import xgboost as xgb
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, \
                            confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score

from config import XGB_PARAMS

parser = OptionParser()

parser.add_option("-p", dest="path", help="Path to the csv.")
(options, args) = parser.parse_args()

if not options.path:
    parser.error("You must pass -p argument")

common_path = os.path.join('logs', 'xgb_classifier')
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

# check how many different labels are in tha target. By default, the configuration
# is set for binary classification. In case there are more than two labels, automatically
# changes the objective function of the booster.
if len(np.bincount(y)) > 2:
    XGB_PARAMS["objective"] = "reg:logistic"

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

# clean up memory
gc.collect()
psutil.virtual_memory()

# set a seed
np.random.seed(42)

# split into trainind and testing data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

folds = StratifiedKFold(n_splits=10, shuffle=False, random_state=44000)
oof = np.zeros((x_train.shape[0], 1))
predictions = np.zeros(len(x_test))

for fold_, (trn_idx, val_idx) in enumerate(folds.split(x_train, y_train)):
    print("Fold {}".format(fold_))
    trn_data = xgb.DMatrix(x_train[trn_idx], label=y_train[trn_idx])
    val_data = xgb.DMatrix(x_train[val_idx], label=y_train[val_idx])

    evallist = [(val_data, 'eval'), (trn_data, 'train')]

    num_round = 5000
    clf = xgb.train(XGB_PARAMS, trn_data, num_round, evallist, verbose_eval=500, early_stopping_rounds=700)
    oof[val_idx] = clf.predict(xgb.DMatrix(x_train[val_idx])).reshape((x_train[val_idx].shape[0], 1))

    predictions += clf.predict(xgb.DMatrix(x_test)) / folds.n_splits

    # save model
    print('Saving model...')
    clf.save_model(os.path.join(common_path, 'model_{}.model'.format(fold_)))

print("CV score: {:<8.5f}".format(roc_auc_score(y_test, predictions)))