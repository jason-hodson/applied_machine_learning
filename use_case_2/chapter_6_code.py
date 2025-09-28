import pandas as pd

#read in data
df = pd.read_csv("use_case_2_cleaned.csv")

#select columns for lagging
df_lagging = df[['Customer ID', 'order_date']]

#import specific date libraries
from datetime import datetime, timedelta

#create new date column for the lagging
df_lagging['order_minus_8'] =  pd.to_datetime(df_lagging['order_date']) - timedelta(days=8)

#rename the date column
df_lagging['max_date'] = df_lagging['order_date']

#drop the old date column name
df_lagging = df_lagging.drop(columns=['order_date'])

#join the data
df_lag_7 = pd.merge(df[['Customer ID', 'order_date']], df_lagging, how = 'inner', on = 'Customer ID')

#filter to only the rows to keep
df_lag_7 = df_lag_7[(df_lag_7['order_date'] > df_lag_7['order_minus_8']) & (df_lag_7['order_date'] < df_lag_7['max_date'])]

#select only the customer and order date
df_lag_7 = df_lag_7[['Customer ID', 'order_date']]

#remove duplicates
df_lag_7 = df_lag_7.drop_duplicates()

#create a column of all 1’s for the target
df_lag_7['target_7'] = 1

#join the lagged data onto the original dataset
df = pd.merge(df, df_lag_7, how = 'left', on = ['Customer ID', 'order_date'])

#fill any blank values with 0
df['target_7'] = df['target_7'].fillna(0)


#select only the customer and order date
df_lagging = df[['Customer ID', 'order_date']]

#create the new date for the 2 week lag
df_lagging['order_minus_15'] =  pd.to_datetime(df_lagging['order_date']) - timedelta(days=15)

#rename the order date column
df_lagging['max_date'] = df_lagging['order_date']

#drop the old column
df_lagging = df_lagging.drop(columns=['order_date'])

#join the data
df_lag_14 = pd.merge(df[['Customer ID', 'order_date']], df_lagging, how = 'inner', on = 'Customer ID')

#keep only the necessary columns
df_lag_14 = df_lag_14[(df_lag_14['order_date'] > df_lag_14['order_minus_15']) & (df_lag_14['order_date'] < df_lag_14['max_date'])]

#select only the customer and order date
df_lag_14 = df_lag_14[['Customer ID', 'order_date']]

#drop the duplicates
df_lag_14 = df_lag_14.drop_duplicates()

#create the target variable column with 1’s
df_lag_14['target_14'] = 1

#join the data to the original dataset
df = pd.merge(df, df_lag_14, how = 'left', on = ['Customer ID', 'order_date'])

#fill the blank target values with 0
df['target_14'] = df['target_14'].fillna(0)


#load necessary libraries for the modeling process
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score

#select the predictors for the model
X = df.drop(columns=['target_7', 'target_14', 'Customer ID', 'timestamp', 'order_date'])

#select the 1 week target column 
y = df['target_7']

#execute the train test split
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3, random_state=42)

#create the model
model = GradientBoostingClassifier(random_state=42)

#fit the model to the data
model.fit(X_train, y_train)

#calculate both the probability and binary predictions
y_pred_proba = model.predict_proba(X_test)[:, 1]
y_pred = model.predict(X_test)

#calculate the ROC AUC and accuracy scores
auc_score = roc_auc_score(y_test, y_pred_proba)
accuracy = accuracy_score(y_test, y_pred)

#print both the ROC AUC and accuracy scores
print(auc_score)
print(accuracy)

#extract feature importance from the model
importances = model.feature_importances_

#create a data frame of the feature importance and column names
feature_importances = pd.DataFrame({'Feature': X_test.columns, 'Importance': importances})

#sort the data frame so most important features are at the top
feature_importances = feature_importances.sort_values('Importance', ascending=False).reset_index(drop=True)

#select only the top 10 important features
feature_importances = feature_importances.head(10)

#create the feature importance plot
feature_importances.plot(x='Feature', y='Importance', kind='bar')

#select only the customer, date, and bill subtotal
df_last_7_lag = df[['Customer ID', 'order_date', 'Bill subtotal']]

#add 7 days to the data
df_last_7_lag['max_date'] = pd.to_datetime(df_last_7_lag['order_date']) + timedelta(days=8)

#select only the customer id and order date
df_last_7_lag_join = df[['Customer ID', 'order_date']]

#rename the columns
df_last_7_lag_join.columns = ['Customer ID', 'base_order_date']


#join the data
df_last_7_lag = pd.merge(df_last_7_lag_join, df_last_7_lag, how = 'left', on = ['Customer ID'])

#keep only the rows that fit the date range
df_last_7_lag = df_last_7_lag[(df_last_7_lag['base_order_date'] > df_last_7_lag['order_date']) & (df_last_7_lag['base_order_date'] < df_last_7_lag['max_date'])]


#aggegrate the data
df_last_7_lag = df_last_7_lag.groupby(['Customer ID', 'base_order_date']).agg(mean=('Bill subtotal', 'mean'), count=('Bill subtotal', 'count')).reset_index()

#rename the columns
df_last_7_lag.columns = ['Customer ID', 'order_date', 'mean_order_subtotal_last_7', 'count_order_last_7']

#join to original dataset
df = pd.merge(df, df_last_7_lag, how = 'left', on = ['Customer ID', 'order_date'])

#fill any blank values with 0
df = df.fillna(0)


#select the columns to be predictors in the model
X = df.drop(columns=['target_7', 'target_14', 'Customer ID', 'timestamp', 'order_date', 'mean_order_subtotal_last_7'])

#select the one week target variable
y = df['target_7']

#execute the train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#create the model
model = GradientBoostingClassifier(random_state=42)

#fit the model to the data
model.fit(X_train, y_train)

#create both the probability and binary predictions
y_pred_proba = model.predict_proba(X_test)[:, 1]
y_pred = model.predict(X_test)

#calculate both the ROC AUC and accuracy scores
auc_score = roc_auc_score(y_test, y_pred_proba)
accuracy = accuracy_score(y_test, y_pred)

#print both the ROC AUC and accuracy scores
print(auc_score)
print(accuracy)


#extract the feature importance from the model
importances = model.feature_importances_

#put the feature importance into a dataframe with the column names
feature_importances = pd.DataFrame({'Feature': X_test.columns, 'Importance': importances})

#sort the values of the dataframe
feature_importances = feature_importances.sort_values('Importance', ascending=False).reset_index(drop=True)

#select only the top 10 features
feature_importances = feature_importances.head(10)

#plot the features in a bar graph
feature_importances.plot(x='Feature', y='Importance', kind='bar')


#select the columns to be predictors in the model
X = df.drop(columns=['target_7', 'target_14', 'Customer ID', 'timestamp', 'order_date'])

#select the one week target variable
y = df['target_7']

#execute the train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#create the model
model = GradientBoostingClassifier(random_state=42)

#fit the model to the data
model.fit(X_train, y_train)

#create both the probability and binary predictions
y_pred_proba = model.predict_proba(X_test)[:, 1]
y_pred = model.predict(X_test)

#calculate both the ROC AUC and accuracy scores
auc_score = roc_auc_score(y_test, y_pred_proba)
accuracy = accuracy_score(y_test, y_pred)

#print both the ROC AUC and accuracy scores
print(auc_score)
print(accuracy)


#create day of the week feature
df['day_of_week'] = pd.to_datetime(df['order_date']).dt.weekday

#select the columns to be predictors in the model
X = df.drop(columns=['target_7', 'target_14', 'Customer ID', 'timestamp', 'order_date'])

#select the one week target variable
y = df['target_7']

#execute the train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#create the model
model = GradientBoostingClassifier(random_state=42)

#fit the model to the data
model.fit(X_train, y_train)

#create both the probability and binary predictions
y_pred_proba = model.predict_proba(X_test)[:, 1]
y_pred = model.predict(X_test)

#calculate both the ROC AUC and accuracy scores
auc_score = roc_auc_score(y_test, y_pred_proba)
accuracy = accuracy_score(y_test, y_pred)

#print both the ROC AUC and accuracy scores
print(auc_score)
print(accuracy)
