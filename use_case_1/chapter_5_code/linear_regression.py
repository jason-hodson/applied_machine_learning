####Linear regression

#read libraries
import pandas as pd
import numpy as np

#create in cleaned data from previous chapter
df_cleaned = pd.read_csv("df_cleaned.csv")

#create an aggregated version of the data
df_data_grouped = df_cleaned.groupby('invoicedate').agg({'Description': 'nunique', 'Customer ID': 'nunique', 'Country': 'nunique', 'Quantity': 'sum', 'Price':'sum'}).reset_index()

#load in data specific libraries
from datetime import datetime
from dateutil.relativedelta import relativedelta

#create date from the invoice timestamp
df_data_grouped['invoicedate'] = pd.to_datetime(df_data_grouped['invoicedate']).dt.date

#add 3 months to the date
df_data_grouped['invoicedate_minus_3'] = df_data_grouped['invoicedate'] + relativedelta(months=3)

#select only the date and price
df_for_model = df_data_grouped[['invoicedate', 'Price']]

#select only needed columns for the model
df_minus_3 = df_data_grouped[['invoicedate_minus_3', 'Description', 'Customer ID', 'Country', 'Quantity', 'Price']]

#create the month from the date
df_minus_3['month'] = pd.to_datetime(df_minus_3['invoicedate_minus_3']).dt.month

#calculate number of days between the date and the first sale
df_minus_3['since_first_sale'] = pd.to_datetime(df_minus_3['invoicedate_minus_3']) - min(pd.to_datetime(df_for_model['invoicedate']))

#convert calculation to a numeric value
df_minus_3['since_first_sale'] = df_minus_3['since_first_sale'].dt.days

#merge lagged data back onto main dataset
df_for_model = pd.merge(
df_for_model, 
df_minus_3, 
how = 'left', 
left_on = 'invoicedate', 
right_on = 'invoicedate_minus_3', 
suffixes = ('','_3months')
)

#remove records where lagged data is blank
df_for_model = df_for_model[pd.notna(df_for_model['invoicedate_minus_3'])]

#calculate the maximum data date
max_data_date = max(df_data_grouped['invoicedate']) - relativedelta(months=3)

#split into training and test data
training_data = df_for_model[df_for_model['invoicedate'] <= max_data_date]
test_data = df_for_model[df_for_model['invoicedate'] >= max_data_date]

#print number of rows and columns for each training and test datasets
print(training_data.shape)
print(test_data.shape)

#select only numeric columns for the training data
training_data = training_data[['Description', 'Customer ID', 'Country', 'month', 'since_first_sale', 'Price_3months', 'Price']]

#select only numeric columns for the test data
test_data = test_data[['Description', 'Customer ID', 'Country', 'month', 'since_first_sale', 'Price_3months', 'Price']]

#create the predictors for the model’s training and test data
X_train = training_data.iloc[:, :-1]
X_test = test_data.iloc[:, :-1]

#create the target for the training and test data
y_train = training_data.iloc[:, -1]
y_test = test_data.iloc[:, -1]

#read in the linear regression and mae libraries
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

#create model
model = LinearRegression()

#fit the model to the data
model.fit(X_train, y_train)

#create the predictions for the test data
y_pred = model.predict(X_test)

#use MAE to measure the accuracy
accuracy = mean_absolute_error(y_test, y_pred)

#print the accuracy results
print(f"Accuracy: {accuracy}")

#print the sum of both the actual target for the test data and the model’s predictions
print(sum(y_test))
print(sum(y_pred))

#create new holiday season feature on both the training and test data
X_train['holiday_season'] = np.where(X_train['month'] >= 9, 1, 0)
X_test['holiday_season'] = np.where(X_test['month'] >= 9, 1, 0)

#read in plotting library
import matplotlib.pyplot as plt

#plot the actual data
plt.plot(test_data['since_first_sale'], y_test, label = "actual")

#plot the predictions
plt.plot(test_data['since_first_sale'], y_pred, label = "prediction")

#show the legend
plt.legend()


#select only the date and price
df_for_model = df_data_grouped[['invoicedate', 'Price']]

#select only specific fields
df_minus_3 = df_data_grouped[['invoicedate_minus_3', 'Description', 'Customer ID', 'Country', 'Quantity', 'Price']]

#create month feature
df_minus_3['month'] = pd.to_datetime(df_minus_3['invoicedate_minus_3']).dt.month

#create days since first sale feature
df_minus_3['since_first_sale'] = pd.to_datetime(df_minus_3['invoicedate_minus_3']) - min(pd.to_datetime(df_for_model['invoicedate']))

#convert days since first sale to numeric value
df_minus_3['since_first_sale'] = df_minus_3['since_first_sale'].dt.days

#join onto original data
df_for_model = pd.merge(df_for_model, df_minus_3, how = 'left', left_on = 'invoicedate', right_on = 'invoicedate_minus_3', suffixes = ('','_3months'))


#add day of the week feature
df_for_model['day_of_week'] = pd.to_datetime(df_for_model['invoicedate']).dt.weekday

#remove values that the lagged data isn’t populated for
df_for_model = df_for_model[pd.notna(df_for_model['invoicedate_minus_3'])]

#calculate the max data date for splitting our model into train and test
max_data_date = max(df_data_grouped['invoicedate']) - relativedelta(months=3)

#split into training and test data
training_data = df_for_model[df_for_model['invoicedate'] <= max_data_date]
test_data = df_for_model[df_for_model['invoicedate'] >= max_data_date]

#select training data fields
training_data = training_data[['Description', 'Customer ID', 'Country', 'month', 'since_first_sale', 'Price_3months', 'day_of_week', 'Price']]

#select test data fields
test_data = test_data[['Description', 'Customer ID', 'Country', 'month', 'since_first_sale', 'Price_3months', 'day_of_week', 'Price']]

#split into X training and X test
X_train = training_data.iloc[:, :-1]
X_test = test_data.iloc[:, :-1]

#split into y training and y test
y_train = training_data.iloc[:, -1]
y_test = test_data.iloc[:, -1]

#create the holiday season feature on both the training and test data
X_train['holiday_season'] = np.where(X_train['month'] >= 9, 1, 0)
X_test['holiday_season'] = np.where(X_test['month'] >= 9, 1, 0)

#load in linear regression and MAE libraries
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

#create linear regression model
model = LinearRegression()

#fit model to the data
model.fit(X_train, y_train)

#create prediction on the test data
y_pred = model.predict(X_test)

#calculate and print the MAE of the test data results
accuracy = mean_absolute_error(y_test, y_pred)
print(f"Accuracy: {accuracy}")

#load in plotting library
import matplotlib.pyplot as plt

#plot both the actual and prediction data on the graph
plt.plot(test_data['since_first_sale'], y_test, label = "actual")
plt.plot(test_data['since_first_sale'], y_pred, label = "prediction")
plt.legend()

#dummy code the day of the week on the training data
train_weekday_dummies = pd.get_dummies(X_train['day_of_week'], prefix='weekday')

#add dummy coded day of the week to the training data
X_train = pd.concat([X_train, train_weekday_dummies], axis = 1)

#dummy code the day fo the week on the test data
test_weekday_dummies = pd.get_dummies(X_test['day_of_week'], prefix='weekday')

#add dummy coded day of the week to the test data
X_test = pd.concat([X_test, test_weekday_dummies], axis = 1)

