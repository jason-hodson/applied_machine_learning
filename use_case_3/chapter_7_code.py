#load in packages
import pandas as pd
import numpy as np

#read the data
df = pd.read_csv("df_summarized")

#sort the values to ensure it's in ascending order
df = df.sort_values(by=['DATE', 'AREA NAME'])

#shift 7 days for each area name independently
df['crime_count_shifted'] = df.groupby('AREA NAME')['crime_count'].shift(-7)


#we found some bad date values we need to remove
df = df[df['DATE'] != '0']

#convert the DATE column to pandas date for ease of use
df['DATE'] = pd.to_datetime(df['DATE'])

#create numeric day of week column
df['day_of_week'] = df['DATE'].dt.dayofweek

#create numeric day of month column
df['day_of_month'] = df['DATE'].dt.day

#create numeric week of year column
df['week_of_year'] = df['DATE'].dt.isocalendar().week

#create numeric month of year column
df['month_of_year'] = df['DATE'].dt.month

#create numeric month of year column
df['year'] = df['DATE'].dt.year

#rolling average of crime from the last 7 days
df['last_7_days'] = df.groupby('AREA NAME')['crime_count'].rolling(window=7, min_periods=1).mean().reset_index(level=0, drop=True)

#rolling average of crime for the last 30 days
df['last_30_days'] = df.groupby('AREA NAME')['crime_count'].rolling(window=30, min_periods=1).mean().reset_index(level=0, drop=True)

#standard deviation for the last 30 days
df['last_30_days_std'] = df.groupby('AREA NAME')['crime_count'].rolling(window=30, min_periods=1).std().reset_index(level=0, drop=True)

#create historical version of data for visualizing data at the end
df_historicals = df[df['crime_count_shifted'].notnull()]

#filter to only null values in the target variable
df = df[df['crime_count_shifted'].isnull()]

#validate the row count
print(df.shape) 

import pickle

#input file name and open file
file_name = 'model.pkl'
with open(file_name, 'rb') as file:
    #read in the model
    model = pickle.load(file)


#drop the columns not in the model as inputs
X = df.drop(columns=['DATE', 'AREA NAME', 'crime_count', 'crime_count_shifted'])

#predict the results
y_pred = model.predict(X)


#group by date for historical data
df_historicals = df_historicals.groupby('DATE')['crime_count'].sum().reset_index()

#add the predictions onto df
df['predictions'] = y_pred

#group by date for the predictions
df_predictions = df.groupby('DATE')['predictions'].sum().reset_index()


import matplotlib.pyplot as plt

#set the size of the plot
plt.figure(figsize=(9, 5))

#plot the predictions
plt.plot(df_predictions['DATE'], df_predictions['predictions'], label = "future predictions")

#plot the actual values
plt.plot(df_historicals['DATE'], df_historicals['crime_count'], label = "actuals")

#show a legend on the plot
plt.legend()

#filter down df_historicals to just 2024
df_historicals = df_historicals[df_historicals['DATE'] >= '2024-01-01']

#sort and view the last 25 records of the dataset
df_historicals.sort_values(by='DATE', ascending = False).head(25)



df = pd.read_csv("df_summarized")

#sort the values to ensure it's in ascending order
df = df.sort_values(by=['DATE', 'AREA NAME'])

#filter out dates after 12/12
df = df[df['DATE'] <= '2024-12-12']

#shift 7 days for each area name independently
df['crime_count_shifted'] = df.groupby('AREA NAME')['crime_count'].shift(-7)

#we found some bad date values we need to remove
df = df[df['DATE'] != '0']

#convert the DATE column to pandas date for ease of use
df['DATE'] = pd.to_datetime(df['DATE'])

#create numeric day of week column
df['day_of_week'] = df['DATE'].dt.dayofweek

#create numeric day of month column
df['day_of_month'] = df['DATE'].dt.day

#create numeric week of year column
df['week_of_year'] = df['DATE'].dt.isocalendar().week

#create numeric month of year column
df['month_of_year'] = df['DATE'].dt.month

#create numeric month of year column
df['year'] = df['DATE'].dt.year

#rolling average of crime from the last 7 days
df['last_7_days'] = df.groupby('AREA NAME')['crime_count'].rolling(window=7, min_periods=1).mean().reset_index(level=0, drop=True)

#rolling average of crime for the last 30 days
df['last_30_days'] = df.groupby('AREA NAME')['crime_count'].rolling(window=30, min_periods=1).mean().reset_index(level=0, drop=True)

#standard deviation for the last 30 days
df['last_30_days_std'] = df.groupby('AREA NAME')['crime_count'].rolling(window=30, min_periods=1).std().reset_index(level=0, drop=True)

#keep historical data for visualization
df_historicals = df[df['crime_count_shifted'].notnull()]

#filter to only null values in the target variable
df = df[df['crime_count_shifted'].isnull()]

#drop the columns not in the model as inputs
X = df.drop(columns=['DATE', 'AREA NAME', 'crime_count', 'crime_count_shifted'])

#predict the results
y_pred = model.predict(X)

#group historical data by date
df_historicals = df_historicals.groupby('DATE')['crime_count'].sum().reset_index()

#add predictions to dataset
df['predictions'] = y_pred

#group predictions by date
df_predictions = df.groupby('DATE')['predictions'].sum().reset_index()

#filter to just 2024 data
df_historicals = df_historicals[df_historicals['DATE'] >= '2024-01-01']

#set plot size
plt.figure(figsize=(9, 5))

#plot the predictions
plt.plot(df_predictions['DATE'], df_predictions['predictions'], label = "future predictions")

#plot the actuals
plt.plot(df_historicals['DATE'], df_historicals['crime_count'], label = "actuals")

#display legend
plt.legend()


