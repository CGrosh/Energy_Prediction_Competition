#%%
import pandas as pd 
import numpy as np 
import pickle

def feature_engine(df):

    prim_use_feats = ['primary_use_Education', 'primary_use_Office', 
                        'primary_use_Entertainmen/public assembly', 'primary_use_Lodging/residential', 
                        'primary_use_Public Services', 'primary_use_Healthcare', 
                        'primary_use_Other', 'primary_use_Parking', 
                        'primary_use_Manufacturing/industrial', 
                        'primary_use_Food sales and service', 'primary_use_Retail', 
                        'primary_use_Warehouse/storage', 'primary_use_Services', 
                        'primary_use_Technology/science', 'primary_use_Utility', 
                        'primary_use_Religious worship']

    df['Year'] = df['timestamp'].str[:4]
    df['Year'] = df['Year'].astype('int64')
    df['building_age'] = df['Year'] - df['year_built']

    df['Month'] = df['timestamp'].str[5:7]
    df['Month'] = df['Month'].astype('int64')
    df['Day'] = df['timestamp'].str[8:10]
    df['Day'] = df['Day'].astype('int64')

    cols = list(df.columns)
    # cols.remove('meter_reading')
    cols.remove('Unnamed: 0')
    cols.remove('Unnamed: 0.1')
    cols.remove('building_id')
    cols.remove('timestamp')
    cols.remove('year_built')
    cols.remove('site_id')
    cols.remove('Year')
    cols.remove('Month')
    cols.remove('row_id')

    data_x = df[cols]
    data_x = pd.get_dummies(data_x, columns=['primary_use'])

    for i in prim_use_feats:
        if i not in data_x.columns:
            data_x[i] = pd.Series(np.zeros(len(data_x)))
    # data_y = df['meter_reading']

    return data_x 

# test = pd.read_csv('test.csv')
# meta = pd.read_csv('building_metadata.csv')
# weather = pd.read_csv('weather_test.csv')

# test_to_meta = pd.merge(test, meta, on='building_id', how='left')
# data = pd.merge(test_to_meta, weather, on=['site_id', 'timestamp'], how='left')
# data.to_csv('Full_test.csv')
# data = pd.read_csv('Full_test.csv')
data = pd.read_csv('Test_Data_1.csv')
#%%

# val = len(data)/10

# indexes = []
# for i in range(10):
#     indexes.append(int(val))
#     val += 4169760

# for i in range(len(indexes)):
#     if i == 0:
#         data[:indexes[i]].to_csv('Test_Data_{}.csv'.format(i))
#         print("Completed First DataFrame")
#     else:
#         data[indexes[i-1]:indexes[i]].to_csv('Test_Data_{}.csv'.format(i))
#         print("Completed Dataframe Number: {}".format(i))



row = data['row_id']

test_data = feature_engine(data)
#%%
# test_data['primary_use_Manufacturing/industrial'] = pd.Series(np.zeros(len(test_data)))
# test_data['primary_use_Technology/science'] = pd.Series(np.zeros(len(test_data)))

impFile = open("ImputerFile", "rb")
imp = pickle.load(impFile)

ScalerFile = open("ScalerFile", "rb")
scaler = pickle.load(ScalerFile)

pcaFile = open("PCAFile", "rb")
pca = pickle.load(pcaFile)

modelFile = open("RandomForestModel", "rb")
model = pickle.load(modelFile)

test_data_pp = imp.transform(test_data)
test_data_pp = scaler.transform(test_data_pp)
test_data_pp = pca.transform(test_data_pp)

pred = model.predict(test_data_pp)

sub = pd.DataFrame({'row_id': row, 'meter_reading': pred})

print(sub.head())



# %%
