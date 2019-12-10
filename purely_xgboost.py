#%%
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
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


def run_it(file, rows):
    data = get_data(file, rows)
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
    cols.remove('building_id')
    cols.remove('timestamp')
    cols.remove('year_built')
    # cols.remove('site_id')
    cols.remove('Year')
    cols.remove('Month')
  
    data_x = pd.get_dummies(df[cols], columns=['primary_use', 'site_id'])
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
    data_x['binned_sqft'] = bin_sqft(data_x)

    data_x['floor_count'] = data_x.groupby('binned_sqft').transform(lambda x: x.fillna(x.mean()))
    del data_x['binned_sqft']

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

x_train_pp, x_test_pp, y_train, y_test = run_it('Final_Data.csv', 10000000)

#%%
# xgb_reg_model = xgb.XGBRegressor(objective='reg:squarederror', colsample_bytree=1, 
#                                 learning_rate=0.4, max_depth=20, 
#                                 alpha=10, n_estimators=20)

n_est = 20
max_depth = 20
alpha = 11
learning_rate = 0.4

# xgb_reg_model = xgb.XGBRFRegressor(objective='reg:squarederror', colsample_bytree=1, 
#                                     learning_rate=learning_rate, min_child_weight=2, 
#                                     max_depth=max_depth, alpha=alpha, n_estimators=n_est, 
#                                     tree_method='hist')

xgb_reg_model = xgb.XGBRFRegressor(objective='reg:squarederror', colsample_bytree=1, 
                                    min_child_weight=2, max_depth=max_depth, 
                                    learning_rate=learning_rate, tree_method='hist', 
                                    n_estimators=n_est, alpha=alpha)

# param_grid = [{
#     "n_estimators": [5, 10, 15, 20], 
#     # "max_depth": [5, 10, 15, 20], 
#     "alpha": np.linspace(10, 15, 2), 
#     # "learning_rate":[0.4, 0.3, 0.5]
# }]

# grid_search = GridSearchCV(xgb_reg_model, param_grid, cv=3)

print("Beginning to Train the Model")
start = time.time()
xgb_reg_model.fit(x_train_pp, y_train)
end = time.time()
print('\n')
if (end-start)>=60:
    print("Training took approx. " + str((end-start)/60) + " minutes")
else:
    print("Training took approx. " + str(end-start) + " seconds")
# params  ={"objective":"reg:squarederror", "colsample_bytree":1, 'learning_rate':0.4, 
#             'max_depth':20, 'alpha':10, 'n_estimators':20}

# xgb_reg = xgb.tra4in(params=params, dtrain=dtrain, num_boost_round=10)
# print(grid_search.best_params_)
preds = xgb_reg_model.predict(x_test_pp)
preds = np.absolute(preds)
print('RMSLE: ', np.sqrt(mean_squared_log_error(y_test, preds)))
print("N_estimators :", n_est)
print("Max_Depth: ", max_depth)
print("Alpha Val: ", alpha)
print("Learning Rate: ", learning_rate)
# %

# %%
