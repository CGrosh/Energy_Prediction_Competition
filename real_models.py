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

def run_it(file, rows, model_type, params):
    data = get_data(file, nrows=rows)
    data_pp = process_data(data, 'MinMax', 0.95)
    return data_pp

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
    # cols.remove('Unnamed: 0.1')
    cols.remove('building_id')
    cols.remove('timestamp')
    cols.remove('year_built')
    cols.remove('site_id')
    cols.remove('Year')
    cols.remove('Month')
    cols.remove('row_id')

    data_x = df[cols]
    data_x = pd.get_dummies(data_x, columns=['primary_use'])


def get_data(filename, row_num):
    df = pd.read_csv(filename, nrows=row_num)
    return df

def process_data(df, scale, pca_level):
    data_x = feature_engine(df)
    data_y = df['meter_reading']

    imp = SimpleImputer(missing_values=np.nan, strategy='mean')

    if scale == 'MinMax':
        scaler = preprocessing.MinMaxScaler()
    else:
        scaler = preprocessing.StandardScaler()
    
    pca = PCA(n_components=pca_level)
    
    x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, 
                                        test_size=0.3, random_state=4)
    
    x_train_pp = imp.fit_transform(x_train)
    x_train_pp = scaler.fit_transform(x_train_pp)
    x_train_pp = imp.transform(x_train)
    x_train_pp = scaler.transform(x_train_pp)
    



