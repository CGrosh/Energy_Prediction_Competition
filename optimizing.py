import dask.dataframe as dd 
from sklearn import linear_model
from sklearn import preprocessing
import numpy as np
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

def run_it(rows):
    data = dd.read_csv('Final_Data.csv', parse_dates=['timestamp']).head(n=rows)
    x_train, x_test, y_train, y_test = process_data(data, 0.95)
    return x_train, x_test, y_train, y_test

def feature_engine(df):
    df['Year'] = df['timestamp'].dt.year 
    df['building_age'] = df['Year'] - df['year_built']
    df['Month'] = df['timestamp'].dt.month 
    df['Day'] = df['timestamp'].dt.day 

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

    data_x = dd.get_dummies(df[cols], columns=['primary_use'])
    return data_x 

def process_data(df, pca_level):
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

x_train_pp, x_test_pp, y_train, y_test = run_it(9000000)

LightGBM = lgb.sklearn.LGBMRegressor(boosting_type='gbdt', n_estimators=20, num_leaves=100,
                                    max_depth=20)

xgb_reg_model = xgb.XGBRegressor(objective='reg:squarederror', colsample_bytree=1, 
                            learning_rate=0.4, max_depth=20, alpha=10, n_estimators=20)

# LightGBM.fit(x_train_pp, y_train)
start = time.time()
xgb_reg_model.fit(x_train_pp, y_train)

# preds = LightGBM.predict(x_test_pp)
preds = xgb_reg_model.predict(x_test_pp)
end = time.time()
print("Time: ", end-start)
preds = np.absolute(preds)
print("RMSLE: ", np.sqrt(mean_squared_log_error(y_test, preds)))
