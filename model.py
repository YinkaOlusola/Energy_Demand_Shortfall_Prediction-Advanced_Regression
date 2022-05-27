"""

    Helper functions for the pretrained model to be used within our API.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.

    Importantly, you will need to modify this file by adding
    your own data preprocessing steps within the `_preprocess_data()`
    function.
    ----------------------------------------------------------------------

    Description: This file contains several functions used to abstract aspects
    of model interaction within the API. This includes loading a model from
    file, data preprocessing, and model prediction.  

"""

# Helper Dependencies
import numpy as np
import pandas as pd
import pickle
import json

def _preprocess_data(data):
    """Private helper function to preprocess data for model prediction.

    NB: If you have utilised feature engineering/selection in order to create
    your final model you will need to define the code here.


    Parameters
    ----------
    data : str
        The data payload received within POST requests sent to our API.

    Returns
    -------
    Pandas DataFrame : <class 'pandas.core.frame.DataFrame'>
        The preprocessed data, ready to be used our model for prediction.
    """
    # Convert the json string to a python dictionary object
    feature_vector_dict = json.loads(data)
    # Load the dictionary as a Pandas DataFrame.
    feature_vector_df = pd.DataFrame.from_dict([feature_vector_dict])

    # ---------------------------------------------------------------
    # NOTE: You will need to swap the lines below for your own data
    # preprocessing methods.
    #
    # The code below is for demonstration purposes only. You will not
    # receive marks for submitting this code in an unchanged state.
    # ---------------------------------------------------------------

    # ----------- Replace this code with your own preprocessing steps --------

    feature_vector_df = feature_vector_df.drop(['Unnamed: 0'], axis = 1)
    feature_vector_df['Valencia_pressure'] = feature_vector_df['Valencia_pressure'].fillna(feature_vector_df['Valencia_pressure'].mean())
    feature_vector_df['time'] = pd.to_datetime(feature_vector_df['time'])

    column_list =[]

    feature_vector_df['Day'] = feature_vector_df['time'].dt.day               # day
    feature_vector_df['Month'] = feature_vector_df['time'].dt.month           # month
    feature_vector_df['Year'] = feature_vector_df['time'].dt.year             # year
    feature_vector_df['hour'] = feature_vector_df['time'].dt.hour             # hour
    feature_vector_df['minute'] = feature_vector_df['time'].dt.minute         # minute
    feature_vector_df['second'] = feature_vector_df['time'].dt.second         # second

    # adding the new features to the dataset 
    column_list = ['time', 'hour', 'Day','Month','Year','minute','second'] + list(feature_vector_df.columns[1:-6])
    feature_vector_df = feature_vector_df[column_list]

    feature_vector_df['Valencia_wind_deg'] = feature_vector_df['Valencia_wind_deg'].str.extract('(\d+)')      # Extracting the digits from the values in the Valencia_wind_deg column
    feature_vector_df['Valencia_wind_deg'] = pd.to_numeric(feature_vector_df['Valencia_wind_deg'])            # Converting the extracted digits from the values in the Valencia_wind_deg column into numbers
    feature_vector_df['Seville_pressure'] = feature_vector_df['Seville_pressure'].str.extract('(\d+)')        # Extracting the digits from the values in the Seville_pressure column into numbers
    feature_vector_df['Seville_pressure'] = pd.to_numeric(feature_vector_df['Seville_pressure'])              # Converting the extracted digits in the Seville_pressure column into numbers into numbers

    # Dropping the noise and features with weak correlation with response variable from the dataset
    feature_vector_df = feature_vector_df.drop(['time', 'Month', 'minute', 'second', 'Madrid_temp_min','Seville_temp_min', 'Bilbao_temp_max','Bilbao_temp_min','Valencia_temp_max','Valencia_temp_min','Barcelona_temp_max','Barcelona_temp_min', 'Madrid_pressure', 'Valencia_pressure', 'Barcelona_weather_id', 'Seville_weather_id', 'Valencia_humidity', 'Bilbao_pressure', 'Madrid_weather_id', 'Valencia_snow_3h', 'Barcelona_rain_3h', 'Madrid_rain_1h', 'Seville_rain_1h', 'Bilbao_snow_3h', 'Seville_rain_3h', 'Barcelona_pressure', 'Seville_wind_speed', 'Barcelona_rain_1h', 'Bilbao_wind_speed', 'Madrid_clouds_all', 'Seville_clouds_all'], axis=1)


    # This conditional statement checks if the response variable is present in the supplied data and the drop it if present
    if 'load_shortfall_3h' in feature_vector_df.columns:
        y = feature_vector_df[['load_shortfall_3h']]
        feature_vector_df = feature_vector_df.drop('load_shortfall_3h',axis=1)

    predict_vector = feature_vector_df
    # ------------------------------------------------------------------------

    return predict_vector

def load_model(path_to_model:str):
    """Adapter function to load our pretrained model into memory.

    Parameters
    ----------
    path_to_model : str
        The relative path to the model weights/schema to load.
        Note that unless another file format is used, this needs to be a
        .pkl file.

    Returns
    -------
    <class: sklearn.estimator>
        The pretrained model loaded into memory.

    """
    return pickle.load(open(path_to_model, 'rb'))


""" You may use this section (above the make_prediction function) of the python script to implement 
    any auxiliary functions required to process your model's artifacts.
"""

def make_prediction(data, model):
    """Prepare request data for model prediction.

    Parameters
    ----------
    data : str
        The data payload received within POST requests sent to our API.
    model : <class: sklearn.estimator>
        An sklearn model object.

    Returns
    -------
    list
        A 1-D python list containing the model prediction.

    """
    # Data preprocessing.
    prep_data = _preprocess_data(data)
    # Perform prediction with model and preprocessed data.
    prediction = model.predict(prep_data)
    # Format as list for output standardisation.
    return prediction[0].tolist()
