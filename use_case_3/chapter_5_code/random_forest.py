#Assumes the data from the decision tree section are still in memory

#load in libraries necessary for the modeling process
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

#split data into what will be used to predict and what is being predicted
X = df.drop(columns=['DATE', 'AREA NAME', 'crime_count'])
y = df['crime_count']

#perform standard train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#create and fit the random forest model
model = RandomForestRegressor(max_depth=17)
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

#sort the values to ensure it's in ascending order
df = df.sort_values(by=['DATE', 'AREA NAME'])

#shift 7 days for each area name independently
df['crime_count_shifted'] = df.groupby('AREA NAME')['crime_count'].shift(-7)

#filter to remove what are now null values for the latest dates of the dataset
df = df[df['crime_count_shifted'].notnull()]

#split data into what will be used to predict and what is being predicted
X = df.drop(columns=['DATE', 'AREA NAME', 'crime_count', 'crime_count_shifted'])
y = df['crime_count_shifted']

#perform standard train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#create and fit the random forest model
model = RandomForestRegressor(max_depth=17)
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


