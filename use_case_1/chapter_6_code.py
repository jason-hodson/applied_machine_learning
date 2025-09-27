#import pandas and numpy
import pandas as pd
import numpy as np

#load in the data
df_cleaned = pd.read_csv("df_cleaned.csv")

#aggregate the data
df_data_grouped = df_cleaned.groupby('invoicedate').agg(
{
'Description': 'nunique', 
'Customer ID': 'nunique', 
'Country': 'nunique', 
'Quantity': 'sum', 
'Price':'sum'
}
).reset_index()

#import date specific libraries
from datetime import datetime
from dateutil.relativedelta import relativedelta

#create date columns
df_data_grouped['invoicedate'] = pd.to_datetime(
df_data_grouped['invoicedate']
).dt.date

#create lagged date column
df_data_grouped['invoicedate_minus_3'] = 
df_data_grouped['invoicedate'] + relativedelta(months=3)

#create data frame with just the date and price
df_for_model = df_data_grouped[['invoicedate', 'Price']]

#create dataframe for lagging
df_minus_3 = df_data_grouped[[
'invoicedate_minus_3', 
'Description', 
'Customer ID', 
'Country', 
'Quantity', 
'Price'
]]

#create month feature
df_minus_3['month'] = pd.to_datetime(
df_minus_3['invoicedate_minus_3']
).dt.month

#create first sale date feature
df_minus_3['since_first_sale'] = pd.to_datetime(
df_minus_3['invoicedate_minus_3']) – 
min(pd.to_datetime(df_for_model['invoicedate']))

#update datatype of the first sale date to be a number
df_minus_3['since_first_sale'] = df_minus_3['since_first_sale'].dt.days

#join the data frames together
df_for_model = pd.merge(
df_for_model, 
df_minus_3,
how = 'left', 
left_on = 'invoicedate', 
right_on = 'invoicedate_minus_3', 
suffixes = ('','_3months')
)

#remove any rows with a blank for the lagged data
df_for_model = df_for_model[pd.notna(df_for_model['invoicedate_minus_3'])]

#create the day of the week feature
df_for_model['day_of_week'] = pd.to_datetime(df_for_model['invoicedate']).dt.weekday


#identify the max date for splitting into the train and test data
max_data_date = max(df_data_grouped['invoicedate']) - relativedelta(months=3)

#create the training and test data
training_data = df_for_model[df_for_model['invoicedate'] <= max_data_date]
test_data = df_for_model[df_for_model['invoicedate'] >= max_data_date]

#print the number of rows and columns for the training and test data
print(training_data.shape)
print(test_data.shape)

#specify the columns for the training data
training_data = training_data[['Description', 'Customer ID', 'Country', 'month', 'since_first_sale', 'Price_3months', 'day_of_week','Price']]

#specify the columns for the test data
test_data = test_data[['Description', 'Customer ID', 'Country', 'month', 'since_first_sale', 'Price_3months', 'day_of_week','Price']]

#select only the columns used as predictors
X_train = training_data.iloc[:, :-1]
X_test = test_data.iloc[:, :-1]

#select only the target column
y_train = training_data.iloc[:, -1]
y_test = test_data.iloc[:, -1]

#create the holiday season feature
X_train['holiday_season'] = np.where(X_train['month'] >= 9, 1, 0)
X_test['holiday_season'] = np.where(X_test['month'] >= 9, 1, 0)

#dummy code the day of the week feature for the training data
train_weekday_dummies = pd.get_dummies(X_train['day_of_week'], prefix='weekday')

#add the dummy coded values onto X_train
X_train = pd.concat([X_train, train_weekday_dummies], axis = 1)

#dummy code the day of the week feature for the test data
test_weekday_dummies = pd.get_dummies(X_test['day_of_week'], prefix='weekday')

#add the dummy coded values onto X_test
X_test = pd.concat([X_test, test_weekday_dummies], axis = 1)


#load in necessary libraries
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

#create the model with the last hyperparameters used in the last chapter
model = GradientBoostingRegressor(n_estimators=100, learning_rate = .01, subsample = .9, random_state=42)

#fit the model to the data
model.fit(X_train, y_train)

#create predictions on the test data
y_pred = model.predict(X_test)

#create predictions on the training data
y_train_pred = model.predict(X_train)

#calculate and print the MAE for the training data
accuracy = mean_absolute_error(y_train, y_train_pred)
print(f" Training Accuracy: {accuracy}")

#calculate and print the MAE for the test data
accuracy = mean_absolute_error(y_test, y_pred)
print(f"Accuracy: {accuracy}")

#import the plotting library
import matplotlib.pyplot as plt

#plot both the actual and prediction data
plt.plot(test_data['since_first_sale'], y_test, label = "actual")
plt.plot(test_data['since_first_sale'], y_pred, label = "prediction")
plt.legend()

#identify the importance of the features
importances = model.feature_importances_

#put feature importance and names of the features into a data frame
feature_importances = pd.DataFrame({'Feature': X_test.columns, 'Importance': importances})

#sort the data frame to show the important features first
feature_importances = feature_importances.sort_values('Importance', ascending=False).reset_index(drop=True)

#plot the feature importance
feature_importances.plot(x='Feature', y='Importance', kind='bar')

#create day of the year feature
df_for_model['day_of_year'] = pd.to_datetime(df_for_model['invoicedate']).dt.dayofyear


#identify the max date for the train test split
max_data_date = max(df_data_grouped['invoicedate']) - relativedelta(months=3)

#create the training and test data
training_data = df_for_model[df_for_model['invoicedate'] <= max_data_date]
test_data = df_for_model[df_for_model['invoicedate'] >= max_data_date]

#print the number of rows and columns for the training and test data
print(training_data.shape)
print(test_data.shape)

#select the columns for the training data
training_data = training_data[['Description', 'Customer ID', 'Country', 'month', 'since_first_sale', 'Price_3months', 'day_of_week','day_of_year','Price']]

#select the columns for the test data
test_data = test_data[['Description', 'Customer ID', 'Country', 'month', 'since_first_sale', 'Price_3months', 'day_of_week','day_of_year','Price']]

#select the predictors to become the X_train and X_test data frames
X_train = training_data.iloc[:, :-1]
X_test = test_data.iloc[:, :-1]

#select the target column for both train and test data
y_train = training_data.iloc[:, -1]
y_test = test_data.iloc[:, -1]

#create the holiday season feature
X_train['holiday_season'] = np.where(X_train['month'] >= 9, 1, 0)
X_test['holiday_season'] = np.where(X_test['month'] >= 9, 1, 0)

#dummy code the weekday feature
train_weekday_dummies = pd.get_dummies(X_train['day_of_week'], prefix='weekday')

#add the dummy coded weekday feature to X_train
X_train = pd.concat([X_train, train_weekday_dummies], axis = 1)

#dummy code the weekday feature
test_weekday_dummies = pd.get_dummies(X_test['day_of_week'], prefix='weekday')

#add the dummy coded weekday feature to X_test
X_test = pd.concat([X_test, test_weekday_dummies], axis = 1)


#load necessary libraries
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

#create the model
model = GradientBoostingRegressor(
n_estimators=100, 
learning_rate = .01, 
subsample = .9, 
random_state=42
)

#fit the model to the data
model.fit(X_train, y_train)

#create predictions for the test data
y_pred = model.predict(X_test)

#create predictions for the training data
y_train_pred = model.predict(X_train)

#calculate and print the MAE for the training data
accuracy = mean_absolute_error(y_train, y_train_pred)
print(f" Training Accuracy: {accuracy}")

#calculate and print the MAE for the test data
accuracy = mean_absolute_error(y_test, y_pred)
print(f"Accuracy: {accuracy}")

#plot the actual and prediction results
import matplotlib.pyplot as plt
plt.plot(test_data['since_first_sale'], y_test, label = "actual")
plt.plot(test_data['since_first_sale'], y_pred, label = "prediction")
plt.legend()


#identify the importance of the features
importances = model.feature_importances_

#put the feature importance values and column names into a data frame
feature_importances = pd.DataFrame(
{'Feature': X_test.columns, 
'Importance': importances}
)

#sort the data frame so the most iomprtant features are at the top
feature_importances = feature_importances.sort_values('Importance', ascending=False).reset_index(drop=True)

#plot the feature importance data
feature_importances.plot(x='Feature', y='Importance', kind='bar')

#create week of the year feature
df_for_model['week_of_year'] = pd.to_datetime(df_for_model['invoicedate']).dt.isocalendar().week


#load necessary libraries
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor

#create grid of hyperparameters to test
param_grid = {
'n_estimators': [100, 300, 500],
'learning_rate': [.01, .05, .1],
'subsample': [.7, .8, .9]
}

#create the model
model = GradientBoostingRegressor(random_state=42)

#set up the grid search
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3,scoring='neg_mean_absolute_error')

#execute the grid search
grid_search.fit(X_train, y_train)

#print the best parameters from the grid search
print("Best parameters:", grid_search.best_params_)

#print the best score from the grid search
print("Best score:", grid_search.best_score_)

#updated grid search parameters
param_grid = {
'n_estimators': [250, 300, 350],
'learning_rate': [.005, .01, .15],
'subsample': [.85, .87, .9, .92]
}

#set up grid search
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5,scoring='neg_mean_absolute_error')

#load in train test split
from sklearn.model_selection import train_test_split

#create the predictors for the model
X = df_for_model[['Description', 'Customer ID', 'Country', 'month', 'since_first_sale', 'Price_3months', 'day_of_week','day_of_year','week_of_year']]

#select the target variable
y = df_for_model['Price']

#create the holiday season feature
X['holiday_season'] = np.where(X['month'] >= 9, 1, 0)

#dummy code the weekday feature
x_weekday_dummies = pd.get_dummies(X['day_of_week'], prefix='weekday')

#add the dummy coded weekday feature on X
X = pd.concat([X, x_weekday_dummies], axis = 1)

#execute the randomized train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

#create model
model = GradientBoostingRegressor(
n_estimators=250, 
learning_rate = .01, 
subsample = .9, 
random_state=42
)

#fit the model to the data
model.fit(X_train, y_train)

#predict the test values
y_pred = model.predict(X_test)

#predict all values
y_pred_all = model.predict(X)

#predict the training values
y_train_pred = model.predict(X_train)

#calculate and print the MAE for the training data
accuracy = mean_absolute_error(y_train, y_train_pred)
print(f" Training Accuracy: {accuracy}")

#calculate and print the MAE for the test data
accuracy = mean_absolute_error(y_test, y_pred)
print(f"Accuracy: {accuracy}")

#plot the actuals and predictions
import matplotlib.pyplot as plt
plt.plot(X['since_first_sale'], y, label = "actual")
plt.plot(X['since_first_sale'], y_pred_all, label = "prediction")
plt.legend()


#import shap library
import shap

#create the SHAP explainer using the last model we created
explainer = shap.TreeExplainer(model)

#create the shap values for the test data
shap_values = explainer.shap_values(X_test)

#show the summary plot
shap.summary_plot(shap_values, X_test)


#load pickle library
import pickle

#create model.pkl file and save the model to it
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)
