#Assumes the data from decision tree model section is still in memory

#load in packages
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

#create the model
model = RandomForestRegressor(n_estimators=100, random_state=42)

#fit the model to the data
model.fit(X_train, y_train)

#use the model to create predictions
y_pred = model.predict(X_test)

#use the model to predict on the training data
y_train_pred = model.predict(X_train)

#calculate the accuracy on the training data
accuracy = mean_absolute_error(y_train, y_train_pred)

#print the accuracy of the training data
print(f" Training Accuracy: {accuracy}")

#calculate the accuracy on the test data
accuracy = mean_absolute_error(y_test, y_pred)

#print the accuracy on the test data
print(f"Accuracy: {accuracy}")

#load plotting library
import matplotlib.pyplot as plt

#plot both the actual and prediction data
plt.plot(test_data['since_first_sale'], y_test, label = "actual")
plt.plot(test_data['since_first_sale'], y_pred, label = "prediction")
plt.legend()


#load in the necessary packages
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

#create the model with additional hyperparameters
model = RandomForestRegressor(
n_estimators=300, 
max_depth = 10, 
min_samples_split = 15, 
random_state=42
)

#fit the model to the data
model.fit(X_train, y_train)

#create prediction on test data
y_pred = model.predict(X_test)

#create prediction on the training data
y_train_pred = model.predict(X_train)

#calculate and print MAE on the training data
accuracy = mean_absolute_error(y_train, y_train_pred)
print(f" Training Accuracy: {accuracy}")

#calculate and print MAE on the test data
accuracy = mean_absolute_error(y_test, y_pred)
print(f"Accuracy: {accuracy}")
