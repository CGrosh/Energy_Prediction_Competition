#%% 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 

data = pd.read_csv('Final_Data.csv', nrows=1000000)

#%%
# print(data['timestamp'].str[:4])
data['Year'] = data['timestamp'].str[:4]
data['Year'] = data['Year'].astype('int64')
data['building_age'] = data['Year'] - data['year_built']

data['Month'] = data['timestamp'].str[5:7]
data['Month'] = data['Month'].astype('int64')
data['Day'] = data['timestamp'].str[8:10]
#%%

data.plot(kind='scatter', x='wind_speed', y='meter_reading', figsize=(20,20))
# plt.rcParams['agg.path.chunksize'] = 1000
plt.show()

# %%
