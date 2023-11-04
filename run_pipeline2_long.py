import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
import xgboost as xgb


from utils import *
from sklearn.metrics import mean_squared_error
from sklearn.multioutput import MultiOutputRegressor

dataloader = LoadData(impute='KNN',load = True)
cv = CV(dataloader)
print('about to train')
x = xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree = 0.3, learning_rate = 0.1)
model = MultiOutputRegressor(x)
predicts = cv.train_loop(model)

cv.save_file('XGB')