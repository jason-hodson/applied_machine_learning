#Assumes continuation and features from the regression section are still in memory

#read in decision tree libraries
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

#create the decision tree model
regressor = DecisionTreeRegressor()

#fit the decision tree model to our training data
regressor.fit(X_train, y_train)

#use the model to predict on our test data
y_pred = regressor.predict(X_test)

#calculate MAE
mae = mean_absolute_error(y_test, y_pred)

#print MAE
print("Mean Absolute Error:", mae)

#sum the actuals and predictions for the test data
print(sum(y_test))
print(sum(y_pred))

#read in library for grid search
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV

#create decision tree model
regressor = DecisionTreeRegressor()

#establish the parameter grid to search over
param_grid = {
    'max_depth': [5, 10, 15],
    'min_samples_split': [5, 10, 20]
}

#fun the grid search function
grid_search = GridSearchCV(estimator=regressor, param_grid=param_grid, cv=3,scoring='neg_mean_absolute_error')

#fit the model to the best results of the grid search
grid_search.fit(X_train, y_train)

#use the model to predict on the test data
y_pred = grid_search.predict(X_test)

#print out the parameters of the best model
print("Best parameters:", grid_search.best_params_)

#print out the best score from the grid search
print("Best score:", grid_search.best_score_)

#calculate MAE
mae = mean_absolute_error(y_test, y_pred)

#print MAE
print("Mean Absolute Error:", mae)

#sum both the actuals and predictions of the test data
print(sum(y_test))
print(sum(y_pred))

#read in plotting library
import matplotlib.pyplot as plt

#plot the actual data
plt.plot(test_data['since_first_sale'], y_test, label = "actual")

#plot the prediction data
plt.plot(test_data['since_first_sale'], y_pred, label = "prediction")

#show the legend
plt.legend()

#updated parameter grid
param_grid = {
    'max_depth': [10, 15],
    'min_samples_split': [5, 10]
}

