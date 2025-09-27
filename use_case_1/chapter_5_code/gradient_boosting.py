#assumes your data is still in memory from previous sections

#load the necessary modeling package
from sklearn.ensemble import GradientBoostingRegressor

#create the model
model = GradientBoostingRegressor(n_estimators=100, random_state=42)

#fit the model to the data
model.fit(X_train, y_train)

#use the model to predict on the test data
y_pred = model.predict(X_test)

#use the model to predict on the training data
y_train_pred = model.predict(X_train)

#calculate and print MAE for the predictions on the training data
accuracy = mean_absolute_error(y_train, y_train_pred)
print(f" Training Accuracy: {accuracy}")

#calculate and print MAE for the predictions on the test data
accuracy = mean_absolute_error(y_test, y_pred)
print(f"Accuracy: {accuracy}")

#update to give the GBM new hyperparameters
model = GradientBoostingRegressor(n_estimators=300, learning_rate = .01, subsample = .75, random_state=42)

#load the necessary libraries 
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor

#create a parameter grid to search through
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

#create model using best hyperparameter options from the grid search
model = GradientBoostingRegressor(n_estimators=100, learning_rate = .01, subsample = .9, random_state=42)

#fit the model to the data
model.fit(X_train, y_train)

#predict the test values
y_pred = model.predict(X_test)

#use the model to pick the training values
y_train_pred = model.predict(X_train)

#calculate and print the MAE for the training predictions
accuracy = mean_absolute_error(y_train, y_train_pred)
print(f" Training Accuracy: {accuracy}")

#calculate and print the MAE for the test predictions
accuracy = mean_absolute_error(y_test, y_pred)
print(f"Accuracy: {accuracy}")

#import the plotting library
import matplotlib.pyplot as plt

#plot the actual and prediction data
plt.plot(test_data['since_first_sale'], y_test, label = "actual")
plt.plot(test_data['since_first_sale'], y_pred, label = "prediction")
plt.legend()

