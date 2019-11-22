import pandas as pd 
import numpy as np 
import tensorflow as tf 
import functools 

# Function to detect if columns are numerical or categorical 
def cat_features(dataframe):
    td = pd.DataFrame({'a': [1,2,3], 'b': [1.0, 2.0, 3.0]})
    return filter(lambda x: not(dataframe[x].dtype in [td['a'].dtype, td['b'].dtype]), list(dataframe))
