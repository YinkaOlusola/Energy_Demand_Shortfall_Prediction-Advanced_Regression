"""

    Simple Script to test the API once deployed

    Author: Explore Data Science Academy.

    ----------------------------------------------------------------------

    Description: This file contains code used to formulate a POST request
    which can be used to develop/debug the Model API once it has been
    deployed.

"""

# Import dependencies
import requests
import pandas as pd
import numpy as np

# Load data from file to send as an API POST request.
# We prepare a DataFrame with the public test set + riders data
# from the Kaggle challenge.
test = pd.read_csv('./data/df_test.csv')


# Convert our DataFrame to a JSON string.
# This step is necessary in order to transmit our data via HTTP/S
feature_vector_json = test.iloc[1].to_json()

# Specify the URL at which the API will be hosted.
# NOTE: When testing your instance of the API on a remote machine
# replace the URL below with its public IP:

# url = 'http://{public-ip-address-of-remote-machine}:5000/api_v0.1'
url = 'http://192.168.0.133:5000/api_v0.1'

# Perform the POST request.
print(f"Sending POST request to web server API at: {url}")
print("")
print(f"Querying API with the following data: \n {test.iloc[1].to_list()}")
print("")
# Here `api_response` represents the response we get from our API
api_response = requests.post(url, json=feature_vector_json)

# Display the prediction result
print("Received POST response:")
print("*"*50)
print(f"API prediction result: {api_response.json()}")
print(f"The response took: {api_response.elapsed.total_seconds()} seconds")
print("*"*50)







#Save the cleaned data and fill out nulls in "Valencia_pressure" df_test
df_test = df_test
df_test['Valencia_pressure'] = df_test['Valencia_pressure'].fillna(df_test['Valencia_pressure'].mode()[0])


df_cleaner(df_test,column ='Valencia_wind_deg')

df_cleaner(df_test,column ='Seville_pressure')

df_extract_number(df_test,column ='Valencia_wind_deg')

df_test['time'] = pd.to_datetime(df_test['time'])
df_test.time


#Repeat the time engineering process for the test Data

"""
We had to convert the time type from an object to a datetime format using the 'astype' method before desampling

"""
df_test['Year']  = df_test['time'].astype('datetime64').dt.year
df_test['Month_of_year']  = df_test['time'].astype('datetime64').dt.month
df_test['Week_of_year'] = df_test['time'].astype('datetime64').dt.isocalendar().week
df_test['Day_of_year']  = df_test['time'].astype('datetime64').dt.dayofyear
df_test['Day_of_month']  = df_test['time'].astype('datetime64').dt.day
df_test['Day_of_week'] = df_test['time'].astype('datetime64').dt.dayofweek
df_test['Hour_of_week'] = ((df_test['time'].astype('datetime64').dt.dayofweek) * 24 + 24) - (24 - df_test['time'].astype('datetime64').dt.hour)
df_test['Hour_of_day']  = df_test['time'].astype('datetime64').dt.hour

Time_df = df_test.iloc[:,[-8,-7,-6,-5,-4,-3,-2,-1]]
plt.figure(figsize=[10,6])
sns.heatmap(Time_df.corr(),annot=True )

# Drop redundant time data from df_test
df_test = df_test.drop(columns=['time', 'Week_of_year','Day_of_year','Hour_of_week', 'Unnamed: 0'])


#Drop redundant columns in df_test 
df_test = df_test.drop(['Barcelona_temp', 'Barcelona_temp_min', 'Bilbao_temp', 'Bilbao_temp_max',
                        'Madrid_temp', 'Madrid_temp_min', 'Seville_temp_min',
                        'Valencia_temp', 'Valencia_temp_min'],axis =1)


df_test = dele(df_test,t="_weather_id")
