import numpy as np
import pandas as pd

from model import create_model, rmsle_cv
from preprocessing import preprocessing
from feature_engineering import feature_engineering
from config.config import train_config

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler

from loss import rmsle

print(train_config["train_path"])
all_data, ntrain, ntest, y_train, test_ID = preprocessing(train_config["train_path"], train_config["test_path"])
train_data, test_data = feature_engineering(all_data, ntrain, ntest)
model = create_model(name=train_config["model_name"])
if train_config["model_name"] in ['lasso', 'enet']:
    model = make_pipeline(RobustScaler(), model)
    
score = rmsle_cv(model, train_config["n_folds"], train_data, y_train)
print(f'{train_config["model_name"]}: {score.mean()}')

model.fit(train_data, y_train)
model_pred = model.predict(train_data)
model_pred_test = np.expm1(model.predict(test_data))
print(f'Model loss (rmsle): {rmsle(y_train, model_pred)}')


sub = pd.DataFrame()
sub['Id'] = test_ID
sub['SalePrice'] = model_pred_test
sub.to_csv('./submissions/submission.csv', index=False)