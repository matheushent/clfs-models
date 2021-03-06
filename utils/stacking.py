"""Core module for stacking related operations"""
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold

import numpy as np

import pickle
import os

class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, base_models, meta_model, n_folds=5):

        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds
   
    # fit the data on clones of the original models
    def fit(self, X, y):

        self.base_models_ = [list() for x in self.base_models]
        self.meta_model_ = clone(self.meta_model)
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=156)
        
        # train cloned base models then create out-of-fold predictions
        # that are needed to train the cloned meta-model
        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))
        for i, model in enumerate(self.base_models):
            for train_index, holdout_index in kfold.split(X, y):
                instance = clone(model)
                self.base_models_[i].append(instance)
                instance.fit(X[train_index], y[train_index])
                y_pred = instance.predict(X[holdout_index])
                out_of_fold_predictions[holdout_index, i] = y_pred
                
        # Now train the cloned  meta-model using the out-of-fold predictions as new feature
        self.meta_model_.fit(out_of_fold_predictions, y)
        return self
   
    # do the predictions of all base models on the test data and use the averaged predictions as 
    # meta-features for the final prediction which is done by the meta-model
    def predict(self, X):

        meta_features = np.column_stack([
            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1) for base_models in self.base_models_
        ])
        return self.meta_model_.predict(meta_features)

    def save(self, path, base_list, meta_name):

        # save base models
        for base_model, base_name in zip(self.base_models_, base_list):
            with open(os.path.join(path, base_name + '.pickle'), 'wb') as f:
                pickle.dump(base_model[0], f)

        # save meta model
        with open(os.path.join(path, meta_name + '.pickle'), 'wb') as f:
            pickle.dump(self.meta_model_, f)

    def load(self, path, base_list, meta_name):

        # load base models
        self.base_models_ = [list() for x in base_list]

        for i, base_name in base_list:
            with open(os.path.join(path, base_name + '.pickle'), 'rb') as f:
                self.base_models_[i].append(pickle.load(f))

        # load meta model
        with open(os.path.join(path, meta_name + '.pickle'), 'rb') as f:
            self.meta_model_ = pickle.load(f)