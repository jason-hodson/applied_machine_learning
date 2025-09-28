#import libraries
import pandas as pd
import numpy as np

#read in data
df_cleaned = pd.read_csv("df_cleaned.csv")

#group the data
df_data_grouped = df_cleaned.groupby('invoicedate').agg({'Description': 'nunique', 'Customer ID': 'nunique', 'Country': 'nunique', 'Quantity': 'sum', 'Price':'sum'}).reset_index()

#import date specific packages
from datetime import datetime
from dateutil.relativedelta import relativedelta

#create date
df_data_grouped['invoicedate'] = pd.to_datetime(df_data_grouped['invoicedate']).dt.date

#add 3 months to the date
df_data_grouped['invoicedate_minus_3'] = df_data_grouped['invoicedate'] + relativedelta(months=3)

#create date range for the future predictions
dates = pd.date_range(df_data_grouped['invoicedate'].max(),df_data_grouped['invoicedate'].max()+relativedelta(months=3),freq='d')

#put data into a data frame
df_for_model = pd.DataFrame(dates, columns=['invoicedate'])

#convert date to a pandas date time column
df_for_model['invoicedate'] = pd.to_datetime(df_for_model['invoicedate'])

#select specific columns
df_minus_3 = df_data_grouped[['invoicedate_minus_3', 'Description', 'Customer ID', 'Country', 'Quantity', 'Price']]

#create month feature
df_minus_3['month'] = pd.to_datetime(df_minus_3['invoicedate_minus_3']).dt.month

#create days from first sale feature
df_minus_3['since_first_sale'] = pd.to_datetime(df_minus_3['invoicedate_minus_3']) – min(pd.to_datetime(df_for_model['invoicedate']))

#change data type of days from first sale feature
df_minus_3['since_first_sale'] = df_minus_3['since_first_sale'].dt.days

#convert date to pandas date time 
df_minus_3['invoicedate_minus_3'] = pd.to_datetime(df_minus_3['invoicedate_minus_3'])

#join the data
df_for_model = pd.merge(df_for_model, df_minus_3,how = 'left', left_on = 'invoicedate', right_on = 'invoicedate_minus_3', suffixes = ('','_3months'))

#remove blank values
df_for_model = df_for_model[pd.notna(df_for_model['invoicedate_minus_3'])]

#create day of week feature
df_for_model['day_of_week'] = pd.to_datetime(df_for_model['invoicedate']).dt.weekday

#create day of the year feature
df_for_model['day_of_year'] = pd.to_datetime(df_for_model['invoicedate']).dt.dayofyear

#create week of the year feature
df_for_model['week_of_year'] = pd.to_datetime(df_for_model['invoicedate']).dt.isocalendar().week

#rename price column
df_for_model['Price_3months'] = df_for_model['Price']

#select columns to be predictors for the model
X = df_for_model[['Description', 'Customer ID', 'Country', 'month', 'since_first_sale', 'Price_3months', 'day_of_week','day_of_year','week_of_year']]

#create holiday season feature
X['holiday_season'] = np.where(X['month'] >= 9, 1, 0)

#dummy code the weekday feature
x_weekday_dummies = pd.get_dummies(X['day_of_week'], prefix='weekday')

#add the dummy coded weekday feature to the dataset
X = pd.concat([X, x_weekday_dummies], axis = 1)

#drop the weekday_5 column
X = X.drop('weekday_5', axis=1)


#load the pickle library
import pickle

#load the model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)


#read in library
from sklearn.ensemble import GradientBoostingRegressor

#generate predictions
predictions = model.predict(X)


#import plotting library
import matplotlib.pyplot as plt

#set the figure size
plt.figure(figsize=(9, 5))

#plot the future values
plt.plot(df_for_model['invoicedate'], predictions, label = "future predictions")

#plot the historical values
plt.plot(df_data_grouped['invoicedate'], df_data_grouped['Price'], label = "actuals")

#display the legend
plt.legend()

