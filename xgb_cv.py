#%%
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.model_selection import (
    train_test_split, 
    KFold, 
    cross_val_score
)
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
from sklearn.ensemble import (
    VotingRegressor, 
    RandomForestRegressor, 
    GradientBoostingRegressor
)
from sklearn.metrics import explained_variance_score
from sklearn.svm import LinearSVR
import xgboost as xgb
from sklearn.decomposition import PCA 
import pickle 
from sklearn.pipeline import Pipeline


def run_it(file, rows, how):
    data = get_data(file, rows)
    if how == 'CV':
        x, y = CV_pre_process(data, 0.95)
        return x,y
    elif how == 'TTSplit':
        x_train, x_test, y_train, y_test = process_data(data, 'MinMax', 0.95)
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
    # cols.remove('building_id')
    cols.remove('timestamp')
    cols.remove('year_built')
    cols.remove('site_id')
    cols.remove('Year')
    cols.remove('Month')
  
    data_x = pd.get_dummies(df[cols], columns=['primary_use', 'meter'])
    data_x['binned_sqft'] = bin_sqft(data_x)
    data_x['floor_count'] = data_x.groupby('binned_sqft')['floor_count'].transform(\
        lambda x: x.fillna(x.mean()))
    del data_x['binned_sqft']

    data_x['building_id_v2'] = data_x['building_id'].astype('category')
    data_x['building_age'] = data_x.groupby('building_id_v2')['building_age'].transform(\
        lambda x: x.fillna(x.mean()))
    del data_x['building_id']
    del data_x['building_id_v2']

    return data_x

def bin_sqft(df):
    vals = df['square_feet'].describe()
    lst = []
    for i in range(len(df)):
        if df['square_feet'][i] >= vals['75%']:
            lst.append('Top')
        elif df['square_feet'][i] < vals['75%'] and df['square_feet'][i] >= vals['50%']:
            lst.append("High")
        elif df['square_feet'][i] < vals['50%'] and df['square_feet'][i] >= vals['25%']:
            lst.append('Middle')
        elif df['square_feet'][i] < vals['25%']:
            lst.append('Low')
    return pd.Series(lst)

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
    print(x_train.columns)
    x_train_pp = PP_Pipeline.fit_transform(x_train)
    x_test_pp = PP_Pipeline.transform(x_test)

    # PipelineFile = open("PipelineFile", "wb")
    # pickle.dump(PP_Pipeline, PipelineFile)
    # PipelineFile.close()

    print('\n')
    print('Completed Preprocessing and Dimensionality Reduction')
    print('\n')

    return x_train_pp, x_test_pp, y_train, y_test

def CV_pre_process(df, pca_level):
    data_x = feature_engine(df)
    data_y = df['meter_reading']

    PP_Pipeline = Pipeline([
        ('Imputer', SimpleImputer(missing_values=np.nan, strategy='mean')), 
        ('Scaler', preprocessing.MinMaxScaler()), 
        ('PCA', PCA(n_components=pca_level)),
    ])

    x_train_pp = PP_Pipeline.fit_transform(data_x)

    print('\n')
    print('Completed Preprocessing and Dimensionality Reduction')
    print('\n')

    return x_train_pp, data_y

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

    # PipelineFile = open("PipelineFile", "wb")
    # pickle.dump(PP_Pipeline, PipelineFile)
    # PipelineFile.close()

    print('\n')
    print('Completed Preprocessing and Dimensionality Reduction')
    print('\n')

    return x_train_pp, data_y

# x_train_pp, x_test_pp, y_train, y_test = run_it('Final_Data.csv', 4000000)
x, y = run_it('Final_Data.csv', 7000000, 'CV')

#%%
# xgb_reg_model = xgb.XGBRegressor(objective='reg:squarederror', colsample_bytree=1, 
#                                 learning_rate=0.4, max_depth=20, 
#                                 alpha=10, n_estimators=20)

n_est = 20
max_depth = 20
alpha = 11
learning_rate = 0.4

xgb_reg_model = xgb.XGBRFRegressor(objective='reg:squarederror', colsample_bytree=1, 
                                    min_child_weight=2, max_depth=max_depth, 
                                    learning_rate=learning_rate, tree_method='hist', 
                                    n_estimators=n_est, alpha=alpha)

kfold = KFold(n_splits=5, shuffle=True, random_state=4)
kfold_scores = cross_val_score(xgb_reg_model, x, y, scoring='neg_mean_squared_log_error', cv=kfold)
kfold_scores = np.absolute(kfold_scores)
print(np.sqrt(kfold_scores.mean()))
# print("Beginning to Train the Model")
# start = time.time()
# xgb_reg_model.fit(x_train_pp, y_train)
# end = time.time()
# print('\n')
# if (end-start)>=60:
#     print("Training took approx. " + str((end-start)/60) + " minutes")
# else:
#     print("Training took approx. " + str(end-start) + " seconds")

# preds = xgb_reg_model.predict(x_test_pp)
# preds = np.absolute(preds)
# print('RMSLE: ', np.sqrt(mean_squared_log_error(y_test, preds)))
# print("N_estimators :", n_est)
# print("Max_Depth: ", max_depth)
# print("Alpha Val: ", alpha)
# print("Learning Rate: ", learning_rate)

# %%
