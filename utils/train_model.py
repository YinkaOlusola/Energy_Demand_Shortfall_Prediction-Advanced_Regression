"""
    Simple file to create a sklearn model for deployment in our API

    Author: Explore Data Science Academy

    Description: This script is responsible for training a simple linear
    regression model which is used within the API for initial demonstration
    purposes.

"""

# Dependencies
import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# Fetch training data and preprocess for modeling
df_train = pd.read_csv('./data/df_train.csv')

df_train = df_train.drop(['Unnamed: 0'], axis = 1)
df_train['Valencia_pressure'] = df_train['Valencia_pressure'].fillna(df_train['Valencia_pressure'].mean())
df_train['time'] = pd.to_datetime(df_train['time'])

column_list =[]

df_train['Day'] = df_train['time'].dt.day               # day
df_train['Month'] = df_train['time'].dt.month           # month
df_train['Year'] = df_train['time'].dt.year             # year
df_train['hour'] = df_train['time'].dt.hour             # hour
df_train['minute'] = df_train['time'].dt.minute         # minute
df_train['second'] = df_train['time'].dt.second         # second

# adding the new features to the dataset 
column_list = ['time', 'hour', 'Day','Month','Year','minute','second'] + list(df_train.columns[1:-6])
df_train = df_train[column_list]

df_train['Valencia_wind_deg'] = df_train['Valencia_wind_deg'].str.extract('(\d+)')      # Extracting the digits from the values in the Valencia_wind_deg column
df_train['Valencia_wind_deg'] = pd.to_numeric(df_train['Valencia_wind_deg'])            # Converting the extracted digits from the values in the Valencia_wind_deg column into numbers
df_train['Seville_pressure'] = df_train['Seville_pressure'].str.extract('(\d+)')        # Extracting the digits from the values in the Seville_pressure column into numbers
df_train['Seville_pressure'] = pd.to_numeric(df_train['Seville_pressure'])              # Converting the extracted digits in the Seville_pressure column into numbers into numbers

# Dropping the noise and features with weak correlation with response variable from the dataset
df_train = df_train.drop(['time', 'Month', 'minute', 'second', 'Madrid_temp_min','Seville_temp_min', 'Bilbao_temp_max','Bilbao_temp_min','Valencia_temp_max','Valencia_temp_min','Barcelona_temp_max','Barcelona_temp_min', 'Madrid_pressure', 'Valencia_pressure', 'Barcelona_weather_id', 'Seville_weather_id', 'Valencia_humidity', 'Bilbao_pressure', 'Madrid_weather_id', 'Valencia_snow_3h', 'Barcelona_rain_3h', 'Madrid_rain_1h', 'Seville_rain_1h', 'Bilbao_snow_3h', 'Seville_rain_3h', 'Barcelona_pressure', 'Seville_wind_speed', 'Barcelona_rain_1h', 'Bilbao_wind_speed', 'Madrid_clouds_all', 'Seville_clouds_all'], axis=1)

y_train = df_train[['load_shortfall_3h']]                     # Extracting the response variable from the df_train data set
X_train = df_train.drop('load_shortfall_3h',axis=1)           # Extracting the feature variables from the df_train data set

# Fit model
lm_regression = LinearRegression(normalize=True)
random_forest = RandomForestRegressor(n_estimators = 100, random_state = 0)
print ("Training Model...")
lm_regression.fit(X_train, y_train)
random_forest.fit(X_train, y_train)


# Pickle model for use within our API
save_path = '../load-shortfall-regression-predict-api/assets/trained-models/load_shortfall_random_forest_regression.pkl'
print (f"Training completed. Saving model to: {save_path}")
pickle.dump(random_forest, open(save_path,'wb'))
