# clfs-models
Pre-molded models for classification

# Pre-requisites

Unless you want to run Xgboost on GPU, just follow the guide.

Assuming you have installed python 3.6-7 installed, activate the virtual environment and run ```pip3 install -r requirements.txt``` to install the dependencies.

# Models

**OBS**: Note there is no feature engineering in the code. You have to implement your own.

## Stacked Regression

The stacked model uses Lasso as meta model and ElasticNet, KernelRidge and GradientBoosting as base models. 

The final predictoin is made averaging the stacked model with a xgboost and lightgbm models. The proportion follows:

```
Stacked: 0.65
Xgboost: 0.2
Lighgbm: 0.15
```

The logs will be saved in ```logs/stacked_regressor``` folder.

## Xgboost Classifier

The xgboost models is made to make categorical predictions. There is no needed to take care with the label encoding; the code does itself.

Pay attetion there is no feature engineering. The only thing made is the label encoding. Any column that has categorical data will be changed to numerical data.

The code itself is made to classify two or more labels. Feel free to change the params in ```config/__init__.py``` file.

The logs will be save in ```logs/xgb_classifier``` folder.

# Upcoming

* Stacked classifier
* Downsampling classification