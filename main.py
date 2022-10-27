from math import sqrt
from os import path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import tensorflow as tf
from matplotlib import dates as mdates
from matplotlib import pyplot as plt
from tensorflow import keras
from tensorflow.compat.v1.keras.backend import set_session

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement = True  # to log device placement (on which device the operation ran)
                                    # (nothing gets printed in Jupyter, only if you run it standalone)
sess = tf.compat.v1.Session(config=config)
set_session(sess) 

rated_power = 2050
cut_in_speed = 4 # filter data with wind speed above cut-in speed
regression_features = ['Pitch_angle','Absolute_wind_direction','humidity','pressure','temp','Wind_speed','Nacelle_angle']
target_label = ['Active_power']


datafldr = "data"
description = "data_description.csv"
static_info_file = "static-information.csv"
static_info = pd.read_csv(path.join(datafldr,static_info_file),sep=";")
data_description = pd.read_csv(path.join(datafldr,description),sep=";")
turbine_names = static_info["Wind_turbine_name"].to_list()

def load_data(turbine_data,turbine,resample=False):
        # Clean up columns - change to long names
    column_names = turbine_data.keys()
    column_names_short = [x.split('_')[0] for x in column_names if "avg" in x]

    column_names_long = []
    column_names_short_avg = []
    for col in column_names_short:
        idx = (data_description['Variable_name']==col)
        if idx.any():
            col_name_new = data_description.loc[idx,"Variable_long_name"].values[0]
            column_names_long.append(col_name_new)
            column_names_short_avg.append(f"{col}_avg")
    turbine_data_long = turbine_data.copy()
    column_rename = dict(zip(column_names_short_avg,column_names_long))
    turbine_data_long =  turbine_data_long.rename(columns=column_rename)

    # Convert to date-time
    turbine_data_long["Date_time"] = pd.to_datetime(turbine_data_long["Date_time"],utc=True)
    
    # Resample to 1h frequency
    if resample:
        turbine_data_long = turbine_data_long.resample("1h",on="Date_time").first()

    # Drop NaN columns
    print(f"Turbine-{turbine}:Length before NaN drop={len(turbine_data_long)}")
    turbine_data_long = turbine_data_long.dropna()
    print(f"Turbine-{turbine}:Length after NaN drop={len(turbine_data_long)}")

    # Drop data when wind speed is 0
    turbine_data_long = turbine_data_long[turbine_data_long["Wind_speed"]>0.1]
    print(f"Turbine-{turbine}:Length after non-zero wind speed selection = {len(turbine_data_long)}")

    return turbine_data_long

def prepare_train_and_test(df,train_duration=4): 
    # train_duration - number of years of continuous data for training
    start_time = pd.to_datetime(df['Date_time'].values[0],utc=True)
    train_end = start_time+np.timedelta64(train_duration*52,'W')

    idx = (df['Date_time']<train_end)
    if not idx.any():
        train_duration = 4
        train_end = start_time+np.timedelta64(train_duration*52,'W')
        idx = (df['Date_time']<train_end)
    df_train = df.loc[idx,:]
    test_idx = df['Date_time']>train_end
    df_test = df.loc[test_idx,:]
    
    df_train = df_train[df_train['Wind_speed']>cut_in_speed]
    df_test = df_test[df_test['Wind_speed']>cut_in_speed]

    df_train = df_train[df_train['Active_power']>10]
    df_test = df_test[df_test['Active_power']>10]

    train_dataset = df_train[regression_features].copy()
    train_labels = df_train[target_label].copy()

    test_dataset = df_test[regression_features].copy()
    test_labels = df_test[target_label].copy()

    x_train = df_train[['Date_time']]
    x_test = df_test[['Date_time']]

    return (train_dataset,train_labels,x_train,test_dataset,test_labels,x_test)

def train_linear_model(train_dataset,train_labels):
    normalizer = tf.keras.layers.Normalization(axis=-1)
    normalizer.adapt(np.array(train_dataset))

    linear_model = tf.keras.Sequential([
        normalizer,
        tf.keras.layers.Dense(units=1)
    ])

    linear_model.compile( optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),loss='mean_absolute_error')
    print("Training a linear model using Tensorflow")
    linear_model.fit(train_dataset, train_labels,epochs=30,verbose=1,validation_split = 0.2)
    return linear_model

def train_dnn_model(train_dataset,train_labels):
    normalizer = tf.keras.layers.Normalization(axis=-1)
    normalizer.adapt(np.array(train_dataset))

    dnn_model = tf.keras.Sequential([
        normalizer,
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1)
    ])

    dnn_model.compile( optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),loss='mean_absolute_error')
    print("Training a linear model using Tensorflow")
    dnn_model.fit(train_dataset, train_labels,epochs=30,verbose=1,validation_split = 0.2)
    return dnn_model

def predict_wind_power(test_dataset,test_labels,model):
    test_predict = model.predict(test_dataset)
    test_predict[test_predict>rated_power] = rated_power
    test_labels['Prediction'] = test_predict
    test_labels['Residuals'] = test_labels['Active_power']-test_labels['Prediction']
    return test_labels

def plot_predictions(test_predict,x_test,turbine,model_name):
    mean_absolute_error = test_predict['Residuals'].abs().mean()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x = x_test['Date_time'], y = test_predict['Active_power'], mode = 'markers',
                            name = 'True Power'))
    fig.add_trace(go.Scatter(x = x_test['Date_time'], y = test_predict['Prediction'], mode = 'markers',
                            name = 'Predicted Power'))

    fig.update_layout(title = f'{model_name} Predictions: Mean Absolute Error {mean_absolute_error:3.2f}W (Turbine-{turbine})',
                    xaxis = dict(title = 'Time'),
                    yaxis = dict(title = 'Power (W)'))
    fig.write_html(path.join('fig',f'Turbine-{turbine}-{model_name}-predictions.html'))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x = x_test['Date_time'], y = test_predict['Residuals'], mode = 'markers',
                            name = 'Residuals'))
    fig.update_layout(title = f'{model_name} Prediction Residuals: Mean Absolute Error {mean_absolute_error:3.2f}W (Turbine-{turbine})',
                    xaxis = dict(title = 'Time'),
                    yaxis = dict(title = 'Power (W)'))
    fig.write_html(path.join('fig',f'Turbine-{turbine}-{model_name}-residuals.html'))

def train_test_predict(turbine_count=0):
    turbine = turbine_names[turbine_count]
    turbine_file = path.join(datafldr,f"{turbine}.csv")
    turbine_data = pd.read_csv(path.join(datafldr,f"{turbine}.csv"))
    # Convert to date-time
    turbine_data["Date_time"] = pd.to_datetime(turbine_data["Date_time"],utc=True)

    turbine_data_long = load_data(turbine_data,turbine,resample=True)

    (train_dataset,train_labels,x_train,test_dataset,test_labels,x_test) = prepare_train_and_test(turbine_data_long)

    #linear_model = train_linear_model(train_dataset,train_labels)
    #test_predict = predict_wind_power(test_dataset,test_labels,model=linear_model)
    #plot_predictions(test_predict,x_test,turbine,model_name='Linear-Model')

    dnn_model = train_dnn_model(train_dataset,train_labels)
    test_predict = predict_wind_power(test_dataset,test_labels,model=dnn_model)
    plot_predictions(test_predict,x_test,turbine,model_name='DNN-Model')

if __name__=='__main__':
    turbine_count = 0
    for turbine_count in range(1):
        train_test_predict(turbine_count)
