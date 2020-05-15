# Check https://xgboost.readthedocs.io/en/latest/parameter.html for more information.
XGB_PARAMS = {
    'subsample': 0.4,
    'process_type':'default',
    'booster': 'gbtree',
    'colsample_bytree': 0.08,
    'eta': 0.001,
    'max_depth': 5,  
    'eval_metric': ['error', 'logloss', 'auc'],
    'tree_method': 'auto',
    'objective': 'binary:logistic', 
    'verbosity': 1
}