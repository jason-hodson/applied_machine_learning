#Assumes the data from the linear regression is still in memory

#read in libraries
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

#split data into what will be used to predict and what is being predicted
X = df.drop(columns=['DATE', 'AREA NAME', 'crime_count'])
y = df['crime_count']

#perform standard train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#create and fit the decision tree model
model = DecisionTreeRegressor(max_depth=3)
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

#read in plotting libraries
import matplotlib.pyplot as plt
from sklearn import tree

plt.figure(figsize=(20, 15))
_ = tree.plot_tree(model,
                   feature_names=list(X_train.columns),
                   class_names=['No', 'Yes'],
                   filled=True,
                   rounded=True,
                   fontsize=10)
plt.show()

#split data into what will be used to predict and what is being predicted
X = df.drop(columns=['DATE', 'AREA NAME', 'crime_count'])
y = df['crime_count']

#perform standard train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#create and fit the decision tree model
model = DecisionTreeRegressor(max_depth=4)
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


#split data into what will be used to predict and what is being predicted
X = df.drop(columns=['DATE', 'AREA NAME', 'crime_count'])
y = df['crime_count']

#perform standard train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#create and fit the decision tree model
model = DecisionTreeRegressor(max_depth=5)
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


#split data into what will be used to predict and what is being predicted
X = df.drop(columns=['DATE', 'AREA NAME', 'crime_count'])
y = df['crime_count']

#perform standard train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#create and fit the decision tree model
model = DecisionTreeRegressor(max_depth=6)
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


#split data into what will be used to predict and what is being predicted
X = df.drop(columns=['DATE', 'AREA NAME', 'crime_count'])
y = df['crime_count']

#perform standard train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#create and fit the decision tree model
model = DecisionTreeRegressor(max_depth=7)
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

