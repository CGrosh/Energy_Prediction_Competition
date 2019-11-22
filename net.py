import numpy as np 
import matplotlib.pyplot as plt
from sklearn import linear_model 
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
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
from sklearn.metrics import explained_variance_score
import tensorflow as tf 

def rmsle(y_pred, y_true):
    y_pred = tf.cast(y_pred, tf.float64)
    y_pred = tf.math.abs(y_pred)
    y_true = tf.cast(y_true, tf.float64)
    y_true = tf.math.abs(y_true)
    y_pred = tf.nn.relu(y_pred)
    return tf.sqrt(tf.reduce_mean(tf.math.squared_difference(tf.math.log1p(y_pred), tf.math.log1p(y_true))))

data = pd.read_csv('Final_Data.csv', nrows=1000000)

#%%
data['Year'] = data['timestamp'].str[:4]
data['Year'] = data['Year'].astype('int64')
data['building_age'] = data['Year'] - data['year_built']

data['Month'] = data['timestamp'].str[5:7]
data['Month'] = data['Month'].astype('int64')
data['Day'] = data['timestamp'].str[8:10]
data['Day'] = data['Day'].astype('int64')
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
cols.remove('Day')

data_x = data[cols]
data_x = pd.get_dummies(data_x, columns=['primary_use'])
data_y = data['meter_reading']

imp = SimpleImputer(missing_values=np.nan, strategy='mean')
scaler = preprocessing.MinMaxScaler()

x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.3, random_state=4)
# print(x_test)
x_train_pp = imp.fit_transform(x_train)
x_train_pp = scaler.fit_transform(x_train_pp)
x_test_pp = imp.fit_transform(x_test)
x_test_pp = scaler.fit_transform(x_test_pp)

y_train = np.c_[y_train]
y_test = np.c_[y_test]

model = tf.keras.Sequential([
    tf.keras.layers.Dense(50, activation='relu', input_shape=[len(data_x.columns)]), 
    tf.keras.layers.Dense(50, activation='relu'), 
    tf.keras.layers.Dense(1)
])

# optimizer = tf.keras.optimizers.RMSprop(0.001)
optimizer = tf.keras.optimizers.RMSprop(lr=1e-4)

model.compile(loss=rmsle, 
            optimizer=optimizer, 
            metrics=['mae', 'mse'])

# print(model.summary())

model.fit(x_train_pp, y_train, epochs=5)