#%%
import pandas as pd 
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
from sklearn.pipeline import Pipeline
import lightgbm as lgb 

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

def run_it(file, rows):
    data = get_data(file, rows)
    x_train, x_test, y_train, y_test = process_data(data, 'MinMax', 0.90)
    return x_train, x_test, y_train, y_test

def feature_engine(df):
    df['Year'] = df['timestamp'].str[:4]
    df['Year'] = df['Year'].astype('int64')
    df['building_age'] = df['Year'] - df['year_built']

    df['Month'] = df['timestamp'].str[5:7]
    df['Month'] = df['Month'].astype('int64')
    df['Day'] = df['timestamp'].str[8:10]
    df['Day'] = df['Day'].astype('int64')

    cols = list(df.columns)
    cols.remove('meter_reading')
    cols.remove('Unnamed: 0')
    cols.remove('building_id')
    cols.remove('timestamp')
    cols.remove('year_built')
    cols.remove('site_id')
    cols.remove('Year')
    cols.remove('Month')
  
    data_x = pd.get_dummies(df[cols], columns=['primary_use'])
    return data_x


def get_data(filename, row_num):
    df = pd.read_csv(filename, nrows=row_num)
    return df

def process_data(df, scale, pca_level):
    data_x = feature_engine(df)
    data_y = df['meter_reading']

    PP_Pipeline = Pipeline([
        ('Imputer', SimpleImputer(missing_values=np.nan, strategy='mean')), 
        ('Scaler', preprocessing.MinMaxScaler()), 
        ('PCA', PCA(n_components=pca_level)),
    ])
    
    x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, 
                                        test_size=0.3, random_state=4)
    
    x_train_pp = PP_Pipeline.fit_transform(x_train)
    x_test_pp = PP_Pipeline.transform(x_test)

    # PipelineFile = open("PipelineFile", "wb")
    # pickle.dump(PP_Pipeline, PipelineFile)
    # PipelineFile.close()

    print('\n')
    print('Completed Preprocessing and Dimensionality Reduction')
    print('\n')

    return x_train_pp, x_test_pp, y_train, y_test

def train_for_production(df, scale, pca_level):
    data_x = feature_engine(df)
    data_y = df['meter_reading']

    PP_Pipeline = Pipeline([
        ('Imputer', SimpleImputer(missing_values=np.nan, strategy='mean')), 
        ('Scaler', preprocessing.MinMaxScaler()), 
        ('PCA', PCA(n_components=pca_level)),
    ])

    print('\n')
    print('The columns being used are: ')
    print(list(data_x))
    print('\n')

    x_train_pp = PP_Pipeline.fit_transform(data_x)


    PipelineFile = open("PipelineFile", "wb")
    pickle.dump(PP_Pipeline, PipelineFile)
    PipelineFile.close()

    print('\n')
    print('Completed Preprocessing and Dimensionality Reduction')
    print('\n')

    return x_train_pp, data_y


# xgb_reg_model = xgb.XGBRegressor(objective='reg:squarederror', colsample_bytree=1, 
#                             learning_rate=0.4, max_depth=20, alpha=10, n_estimators=20)

# RF_Regressor = RandomForestRegressor(random_state=1, n_estimators=15)

# Voter = VotingRegressor(estimators=[('XGB', xgb_reg_model), ('RF', RF_Regressor)])



x_train_pp, x_test_pp, y_train, y_test = run_it('Final_Data.csv', 10000000)


#%%

# LightGBM = lgb.sklearn.LGBMRegressor(boosting_type='gbdt', n_estimators=20, 
#                                     num_leaves=100, max_depth=-1, 
#                                     min_child_samples=50, learning_rate=0.3,
#                                     min_child_weight=10)

# LightGBM.fit(
#     x_train_pp, 
#     y_train, 
#     eval_set=[(x_test_pp, y_test)], 
#     eval_metric=rmsle_light, 
#     # verbose=False,
# )
# LightGBM.fit(x_train_p-p, y_train)

# leaves_vals = [100, 300, 400, 500, 1000]
# depth_vals = [10, 15, 20, 30]
# data_leaves = []

# start = time.time()
# xgb_reg_model.fit(x_train_pp, y_train)
# Voter.fit(x_train_pp, y_train)

# for leaf in range(len(leaves_vals)):
#     for depth in depth_vals:
#         light_mod = lgb.sklearn.LGBMRegressor(boosting_type='gbdt', 
#                                             n_estimators=20, 
#                                             num_leaves=leaves_vals[leaf], 
#                                             max_depth=depth)
#         start = time.time()
#         light_mod.fit(x_train_pp, y_train)
#         print('\n')
#         print('Finished Training')
#         print('\n')

#         # preds = xgb_reg_model.predict(x_test_pp)
#         preds = light_mod.predict(x_test_pp)
#         preds = np.absolute(preds)
#         end = time.time()
#         print('Time: ', end-start)
#         print("RMSLE: ", np.sqrt(mean_squared_log_error(y_test, preds)))
#         print("Number of Leaves: ", leaves_vals[leaf])
#         print("Max Depth of Model: ", depth)


# print('\n')
# print('Finished Training')
# print('\n')
preds = model_try.predict(x_test_pp)
preds = np.absolute(preds)
print("RMSLE: ", np.sqrt(mean_squared_log_error(y_test, preds)))
# # preds = xgb_reg_model.predict(x_test_pp)
# preds = LightGBM.predict(x_test_pp)
# preds = np.absolute(preds)
# # end = time.time()
# # print('Time: ', end-start)
# print("RMSLE: ", np.sqrt(mean_squared_log_error(y_test, preds)))


    





# %%
