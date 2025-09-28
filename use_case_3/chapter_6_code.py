import pandas as pd

#read in summarized data
df = pd.read_csv("df_summarized")

#sort the values to ensure it's in ascending order
df = df.sort_values(by=['DATE', 'AREA NAME'])

#shift 7 days for each area name independently
df['crime_count_shifted'] = df.groupby('AREA NAME')['crime_count'].shift(7)

#filter to remove what are now null values for the earliest dates of the dataset
df = df[df['crime_count_shifted'].notnull()]

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

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

#split data into what will be used to predict and what is being predicted
X = df.drop(columns=['DATE', 'AREA NAME', 'crime_count', 'crime_count_shifted'])
y = df['crime_count_shifted']

#perform standard train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#create and fit the GBM model
model = GradientBoostingRegressor(
  learning_rate = 0.05, 
  max_depth = 7, 
  n_estimators = 100, 
  subsample = 0.8
)
model.fit(X_train, y_train)

#predict with the modeling using the test data
y_pred = model.predict(X_test)

#calculate the mean absolute error
mae = mean_absolute_error(y_test, y_pred)

#calculate the mean absolute error divided by average of the target
mae_by_avg_target = mae / y_test.mean() 

#print both results
print(mae)
print(mae_by_avg_target)

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#get feature importances from the trained model
importances = model.feature_importances_
feature_names = X.columns

#create a DataFrame for easy sorting and plotting
feat_imp_df = pd.DataFrame({'Feature': feature_names,'Importance': importances}).sort_values(by='Importance', ascending=False)

#filter to top 10
feat_imp_df = feat_imp_df.head(10)

#plot
plt.figure(figsize=(10, 6))
plt.barh(feat_imp_df['Feature'], feat_imp_df['Importance'], color='skyblue')
plt.xlabel('Feature Importance')
plt.title('Gradient Boosting Feature Importances')
plt.gca().invert_yaxis()  # Highest importance at the top
plt.tight_layout()
plt.show()


#create rolling 7 days feature
df['last_7_days'] = df.groupby('AREA NAME')['crime_count'].rolling(window=7, min_periods=1).mean().reset_index(level=0, drop=True)



#split data into what will be used to predict and what is being predicted
X = df.drop(columns=['DATE', 'AREA NAME', 'crime_count', 'crime_count_shifted'])
y = df['crime_count_shifted']

#perform standard train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#create and fit the GBM model
model = GradientBoostingRegressor(
  learning_rate = 0.05, 
  max_depth = 7, 
  n_estimators = 100, 
  subsample = 0.8
)
model.fit(X_train, y_train)

#predict with the modeling using the test data
y_pred = model.predict(X_test)

#calculate the mean absolute error
mae = mean_absolute_error(y_test, y_pred)

#calculate the mean absolute error divided by average of the target
mae_by_avg_target = mae / y_test.mean() 

#print both results
print(mae)
print(mae_by_avg_target)


#check the results
df[df['AREA NAME'] == 'Central'][['DATE', 'crime_count', 'last_7_days']].head(14)


#create 30 day lag feature
df['last_30_days'] = df.groupby('AREA NAME')['crime_count'].rolling(window=30, min_periods=1).mean().reset_index(level=0, drop=True)


#create 30 day standard deviation feature
df['last_30_days_std'] = df.groupby('AREA NAME')['crime_count'].rolling(window=30, min_periods=1).std().reset_index(level=0, drop=True)


#filter out blank values
df = df[df['last_30_days_std'].notna()]


#split data into what will be used to predict and what is being predicted
X = df.drop(columns=['DATE', 'AREA NAME', 'crime_count', 'crime_count_shifted'])
y = df['crime_count_shifted']

#perform standard train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#create and fit the GBM model
model = GradientBoostingRegressor(
  learning_rate = 0.02, 
  max_depth = 10, 
  n_estimators = 300, 
  subsample = 1
)
model.fit(X_train, y_train)

#predict with the modeling using the test data
y_pred = model.predict(X_test)

#calculate the mean absolute error
mae = mean_absolute_error(y_test, y_pred)

#calculate the mean absolute error divided by average of the target
mae_by_avg_target = mae / y_test.mean() 

#print both results
print(mae)
print(mae_by_avg_target)


import numpy as np

#subtract the arrays
error = np.subtract(y_test, y_pred)

#calculate the mean of the differences (error)
error.mean()


#package specific for saving in .pkl format
import pickle

#create file name and open the file
file_name = 'model.pkl'
with open(file_name, 'wb') as file:
    
    #save the model object to the file name
    pickle.dump(model, file)

