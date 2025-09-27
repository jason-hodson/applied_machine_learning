#### Logistic Regression

#read in pandas
import pandas as pd

#read in data from previous chapter
df = pd.read_csv("use_case_2_cleaned.csv")

#create dataset for lagging
df_lagging = df[['Customer ID', 'order_date']]

#preview data
df_lagging.head()

#read in date specific libraries
from datetime import datetime, timedelta

#create new date with difference of 8 days
df_lagging['order_minus_8'] =  pd.to_datetime(df_lagging['order_date']) - timedelta(days=8)

#preview the data
df_lagging.head()

#identify create new column called max_date
df_lagging['max_date'] = df_lagging['order_date']

#drop the order_date column
df_lagging = df_lagging.drop(columns=['order_date'])

#join lagged data onto the original data
df_lag_7 = pd.merge(df[['Customer ID', 'order_date']], df_lagging, how = 'inner', on = 'Customer ID')

#print number of columns and rows for both df and df_lag_7
print(df.shape)
print(df_lag_7.shape)

#filter the data to ensure only the necessary rows are matched
df_lag_7 = df_lag_7[(df_lag_7['order_date'] > df_lag_7['order_minus_8']) & (df_lag_7['order_date'] < df_lag_7['max_date'])]

#show the number of columns and rows in the data
df_lag_7.shape

#select only the customer id and order date fields
df_lag_7 = df_lag_7[['Customer ID', 'order_date']]

#drop duplicates
df_lag_7 = df_lag_7.drop_duplicates()

#create dummy column of all 1 values
df_lag_7['target_7'] = 1

#join the data to the original dataset after correction to the lagged dataset
df = pd.merge(df, df_lag_7, how = 'left', on = ['Customer ID', 'order_date'])

#fill in any blanks with 0
df['target_7'] = df['target_7'].fillna(0)

#select only the customer id and order date fields
df_lagging = df[['Customer ID', 'order_date']]

#create lag for 2 weeks column
df_lagging['order_minus_15'] =  pd.to_datetime(df_lagging['order_date']) - timedelta(days=15)

#create max_date column from order_date
df_lagging['max_date'] = df_lagging['order_date']

#drop the originally named order_date column
df_lagging = df_lagging.drop(columns=['order_date'])

#join the 14 day lag with the lagged dataset
df_lag_14 = pd.merge(df[['Customer ID', 'order_date']], df_lagging, how = 'inner', on = 'Customer ID')

#select only rows that fit the desired criteria
df_lag_14 = df_lag_14[(df_lag_14['order_date'] > df_lag_14['order_minus_15']) & (df_lag_14['order_date'] < df_lag_14['max_date'])]

#select only the customer ID and order date
df_lag_14 = df_lag_14[['Customer ID', 'order_date']]

#drop the duplicates
df_lag_14 = df_lag_14.drop_duplicates()

#create dummy column of all 1’s
df_lag_14['target_14'] = 1

#join onto original dataset
df = pd.merge(df, df_lag_14,how = 'left', on = ['Customer ID', 'order_date'])

#fill the target field with 0’s
df['target_14'] = df['target_14'].fillna(0)

#select only desired columns
X = df[[
       'Total', 'KPT duration (minutes)', 'Rider wait time (minutes)', 'DistanceNumeric', 'Restaurant name_Masala Junction',
       'Restaurant name_Swaad', 'Restaurant name_Tandoori Junction',
       'Restaurant name_The Chicken Junction', 'Subzone_Chittaranjan Park',
       'Subzone_DLF Phase 1', 'Subzone_Greater Kailash 2 (GK2)',
       'Subzone_Sector 135', 'Subzone_Sector 4', 'Subzone_Shahdara',
       'Subzone_Sikandarpur', 'Subzone_Vasant Kunj',
       'Cancellation / Rejection reason_Cancelled by Customer',
       'Cancellation / Rejection reason_Cancelled by Zomato',
       'Cancellation / Rejection reason_Items out of stock',
       'Cancellation / Rejection reason_Kitchen is full',
       'Cancellation / Rejection reason_Merchant device issue'
]]

#select target_7 as the target variable
y = df['target_7']


#read in necessary libraries for the modeling process
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score

#split the data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#create logistic regression model
model = LogisticRegression(solver='liblinear', random_state=42)

#fit the model to the data
model.fit(X_train, y_train)

#use the model to predict on the test data
y_pred = model.predict(X_test)

#calculate and print the accuracy score
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

#calculate and print out the ROC AUC score
y_pred_proba = model.predict_proba(X_test)[:, 1]
auc_score = roc_auc_score(y_test, y_pred_proba)
print(f"ROC AUC Score: {auc_score:.4f}")

#select y as the 2 week target
y = df['target_14']

#execute the train test split
X_train, X_test, y_train, y_test = train_test_split(
X, 
y, 
test_size=0.3, 
random_state=42)

#create logistic regression model
model = LogisticRegression(solver='liblinear', random_state=42)

#fit the model to the data
model.fit(X_train, y_train)

#use the model to predict on the test data
y_pred = model.predict(X_test)

#calculate and print the accuracy metric
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

#calculate and print the roc auc metric
y_pred_proba = model.predict_proba(X_test)[:, 1]
auc_score = roc_auc_score(y_test, y_pred_proba)
print(f"ROC AUC Score: {auc_score:.4f}")

