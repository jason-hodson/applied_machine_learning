#assumes the data previous the previous chapters are still in memory

#import necessary packages for the modeling process
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

#split data into what will be used to predict and what is being predicted
X = df.drop(columns=['DATE', 'AREA NAME', 'crime_count', 'crime_count_shifted'])
y = df['crime_count_shifted']

#perform standard train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#create and fit the GBM model
model = GradientBoostingRegressor(max_depth=7)
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
X = df.drop(columns=['DATE', 'AREA NAME', 'crime_count', 'crime_count_shifted'])
y = df['crime_count_shifted']

#perform standard train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#create and fit the GBM model
model = GradientBoostingRegressor(n_estimators = 300, max_depth=7)
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


from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error

#split data into what will be used to predict and what is being predicted
X = df.drop(columns=['DATE', 'AREA NAME', 'crime_count', 'crime_count_shifted'])
y = df['crime_count_shifted']

#perform standard train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#create the model with random state
model = GradientBoostingRegressor(random_state=42)

#define the hyperparameter grid
param_grid = {
    'n_estimators': [100, 300],
    'max_depth': [3, 7],
    'learning_rate': [0.05, 0.2],
    'subsample': [0.8, 1.0]
}

#set up GridSearchCV
grid_search = GridSearchCV(estimator= model,param_grid=param_grid,scoring='neg_mean_absolute_error',cv=3,n_jobs=-1,verbose=1)

#fit the grid search
grid_search.fit(X_train, y_train)

#best model from grid search
best_model = grid_search.best_estimator_

#predict and evaluate
y_pred = best_model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mae_by_avg_target = mae / y_test.mean()

#output results
print("Best Parameters:", grid_search.best_params_)
print("MAE:", mae)
print("MAE / Avg Target:", mae_by_avg_target)

