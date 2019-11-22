#%%import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from sklearn import linear_model 
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_squared_error, mean_squared_log_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import explained_variance_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.impute import SimpleImputer
from util import *
import time 
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import VotingRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import explained_variance_score
from sklearn.svm import LinearSVR
import xgboost as xgb
from sklearn.decomposition import PCA 
import pickle 

def get_metrics(y_true, y_preds):
    print('R2 Score: ' + str(r2_score(y_true, y_preds)))
    print('Explained Variance Score: ' + str(explained_variance_score(y_true, y_preds)))
    print('Median Absolute Error: ' + str(median_absolute_error(y_true, y_preds)))
    print('Mean Squared Error: ' + str(mean_squared_error(y_true, y_preds)))

def rmsle(predictions, dmat):
    labels = dmat.get_label()
    predictions = np.abs(predictions)
    diffs = np.log(predictions+1) - np.log(labels+1)
    squared_diffs = np.square(diffs)
    avg = np.mean(squared_diffs)
    return ('RMSLE', np.sqrt(avg))

# def picklePut(file, object):
#     FileObj = open(file, "wb")
#     pickle.dump(object, FileObj)
#     FileObj.close()

# def picklePull(file):

# train = pd.read_csv('train.csv')
# meta = pd.read_csv('building_metadata.csv')
# weather = pd.read_csv('weather_train.csv')

# train_to_meta = pd.merge(train, meta, on = 'building_id', how='left')
# data = pd.merge(train_to_meta, weather, on = ['site_id', 'timestamp'], how='left')
# data.to_csv('Final_Data.csv')

data = pd.read_csv('Final_Data.csv', nrows=4000000)
#%%
data['Year'] = data['timestamp'].str[:4]
data['Year'] = data['Year'].astype('int64')
data['building_age'] = data['Year'] - data['year_built']

data['Month'] = data['timestamp'].str[5:7]
data['Month'] = data['Month'].astype('int64')
data['Day'] = data['timestamp'].str[8:10]
data['Day'] = data['Day'].astype('int64')

#%%

# data.plot(kind='scatter', x='Day', y='meter_reading')
# plt.show()
#%%
# data.describe()

cols = list(data.columns)
cols.remove('meter_reading')
cols.remove('Unnamed: 0')
cols.remove('building_id')
cols.remove('timestamp')
cols.remove('year_built')
cols.remove('site_id')
cols.remove('Year')
cols.remove('Month')
# cols.remove('Day')

data_x = data[cols]
data_x = pd.get_dummies(data_x, columns=['primary_use'])
data_y = data['meter_reading']

imp = SimpleImputer(missing_values=np.nan, strategy='mean')
scaler = preprocessing.MinMaxScaler()

x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, 
                                    test_size=0.3, random_state=4)
# print(x_test)
x_train_pp = imp.fit_transform(x_train)
x_train_pp = scaler.fit_transform(x_train_pp)
x_test_pp = imp.fit_transform(x_test)
x_test_pp = scaler.fit_transform(x_test_pp)
dtrain = xgb.DMatrix(x_train_pp, label=y_train)

# ScaleFile = open("ScalerFile", "wb")
# pickle.dump(scaler, ScaleFile)
# ScaleFile.close()

# ImputeFile = open("ImputerFile", "wb")
# pickle.dump(imp, ImputeFile)
# ImputeFile.close()

pca = PCA(n_components=0.95)
x_train_pp = pca.fit_transform(x_train_pp)
x_test_pp = pca.transform(x_test_pp)

# PCAFile = open("PCAFile", "wb")
# pickle.dump(pca, PCAFile)
# PCAFile.close()
print('\n')
print('Completed Preprocessing and Dimensionality Reduction')
print('\n')
# Lin_reg_model = linear_model.LinearRegression()
# model = DecisionTreeRegressor(max_depth=10, min_samples_leaf=5)
xgb_reg_model = xgb.XGBRegressor(objective='reg:squarederror', colsample_bytree=1, 
                        learning_rate=0.4, max_depth=20, alpha=10, n_estimators=20)

# RF_reg_model = RandomForestRegressor(random_state=1, n_estimators=10)

# Voter = VotingRegressor(estimators=[('Linear', Lin_reg_model), 
#                                     ('RandomForest', RF_reg_model), 
#                                     ('XGBoost', xgb_reg_model)])

# Voter = Voter.fit(x_train_pp, y_train)

# params  ={"objective":"reg:linear", "colsample_bytree":0.3, 'learning_rate':0.1, 
#             'max_depth':5, 'alpha':10}

# xg_reg = xgb.train(params=params, dtrain=dtrain, num_boost_round=10)

# xgb.plot_importance(xg_reg)
# plt.rcParams['figure.figsize'] = [5, 5]
# # plt.rcParams['font.size'] = 55
# plt.show()

# cv_results = xgb.cv(dtrain=dtrain, params=params, nfold=5, num_boost_round=50, 
#                     early_stopping_rounds=10, feval = rmsle, as_pandas=True, seed=123)
# model = LinearSVR(epsilon=1.5)
# model = linear_model.Lasso(alpha=1, normalize=True)
# k_fold = KFold(n_splits=5, shuffle=True, random_state=4) # This is 5-fold CV
# k_fold_scores = cross_val_score(model, x_train_pp, data_y, scoring='mean_squared_log_error', cv=k_fold)
# print(k_fold_scores)
# model = linear_model.Ridge(alpha=5, solver='svd')
# model = linear_model.SGDRegressor()
# y_train = np.c_[y_train]
# y_test = np.c_[y_test]
#%%
# RF_reg_model.fit(x_train_pp, y_train)

# RandomForestfile = open("RandomForestModel", "wb")
# pickle.dump(RF_reg_model, RandomForestfile)
# RandomForestfile`.close()

xgb_reg_model.fit(x_train_pp, y_train)
print('\n')
print('Finished Training')
print('\n')
# pred = RF_reg_model.predict(x_test_pp)
pred = xgb_reg_model.predict(x_test_pp)
# # #%%
# get_metrics(y_test, pred)
pred = np.absolute(pred)

print("RMSLE: ", np.sqrt(mean_squared_log_error(y_test, pred)))


#%%
# fig, ax = plt.subplots(figsize=(30, 30))
# xgb.plot_tree(xgb_reg_model, num_trees=1)
# plt.show()



# %%
