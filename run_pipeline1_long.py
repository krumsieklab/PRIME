#we will be running my basic pipeline to load and predict data, we will try SVMs, Random Forests, and potentially XGBoost

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os

from utils import *
print('load data')
dataloder = LoadData(impute='KNN')
m = dataloder.m
p = dataloder.p
p_cols = dataloder.p_cols
p_dict = dataloder.p_dict
cv = CV(m,p,p_cols,p_dict)
fold_list = cv.folds(n = 5, random_state = 42)

# from sklearn.ensemble import RandomForestRegressor
# model = RandomForestRegressor(n_estimators=10, random_state=420)
# predicts = cv.train_loop(model) #simply utilizing the same folds, but it writes it over I'm quite sure!

# #now save it
# cv.save_file('RF')

# print('done with RF')

from sklearn.svm import SVR
svr = SVR(kernel='rbf', gamma='auto', C=1.0, epsilon=0.1, max_iter=30)
from sklearn.multioutput import MultiOutputRegressor
model = MultiOutputRegressor(svr)
predicts = cv.train_loop(model)

cv.save_file('SVR')