#Assumes the data from the decision tree section is still in memory

#load in the train test split library
from sklearn.model_selection import train_test_split

#create the predictors for the model
X = df.drop(columns=['target_7', 'target_14', 'Customer ID', 'timestamp', 'order_date'])

#use the 1 week variable lag as the target
y = df['target_7']

#execute train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#load in libraries for the modeling process
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score

#create the random forest model
model = RandomForestClassifier(random_state=42)

#fit the model to the data
model.fit(X_train, y_train)

#create both the binary and probability predictions
y_pred_proba = model.predict_proba(X_test)[:, 1]
y_pred = model.predict(X_test)

#calculate both the ROC AUC and accuracy scores
auc_score = roc_auc_score(y_test, y_pred_proba)
accuracy = accuracy_score(y_test, y_pred)

#print both the ROC AUC and accuracy scores
print(auc_score)
print(accuracy)

#create the model with specific hyperparameter
model = RandomForestClassifier(max_depth = 3, min_samples_split = 20, random_state=42)

#fit the model to the data
model.fit(X_train, y_train)

#create both the binary and probability prediction
y_pred_proba = model.predict_proba(X_test)[:, 1]
y_pred = model.predict(X_test)

#calculate both the ROC AUC and accuracy score
auc_score = roc_auc_score(y_test, y_pred_proba)
accuracy = accuracy_score(y_test, y_pred)

#print both the ROC AUC and accuracy score
print(auc_score)
print(accuracy)

#adjust the hyperparameters to let the model go deeper into the data
model = RandomForestClassifier(
max_depth = 5, 
min_samples_split = 10, 
random_state=42
)

#fit the model to the data
model.fit(X_train, y_train)

#calculate both the binary and probability predictions
y_pred_proba = model.predict_proba(X_test)[:, 1]
y_pred = model.predict(X_test)

#calculate both the ROC AUC and accuracy scores
auc_score = roc_auc_score(y_test, y_pred_proba)
accuracy = accuracy_score(y_test, y_pred)

#print both the ROC AUC and accuracy scores
print(auc_score)
print(accuracy)

#adjust the hyperparameters
model = RandomForestClassifier(
n_estimators = 500, 
max_depth = 5, 
min_samples_split = 10, 
random_state=42
)

#fit the model to the data
model.fit(X_train, y_train)

#calculate both the probability and binary predictions
y_pred_proba = model.predict_proba(X_test)[:, 1]
y_pred = model.predict(X_test)

#calculate both the ROC AUC and accuracy scores
auc_score = roc_auc_score(y_test, y_pred_proba)
accuracy = accuracy_score(y_test, y_pred)

#print both the ROC AUC and accuracy scores
print(auc_score)
print(accuracy)

#load in the grid search library
from sklearn.model_selection import GridSearchCV

#create a paramater grid to search over
param_grid = {
    'n_estimators': [100, 300, 500],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5, 10]
}

#create the model
model = RandomForestClassifier(random_state=42)

#execute the grid search
grid_search = GridSearchCV(
estimator=model, 
param_grid=param_grid, 
cv=5, 
scoring='roc_auc', 
n_jobs=-1, 
verbose=1
)

#fit the model to the best parameters
grid_search.fit(X_train, y_train)

#identify the best model’s parameters and print them
best_model = grid_search.best_estimator_
print("Best Parameters:", grid_search.best_params_)

#use the best parameters from the grid search to create model
model = RandomForestClassifier(
n_estimators = 500, 
max_depth = 10, 
min_samples_split = 5, 
random_state=42
)

#fit the model to the data
model.fit(X_train, y_train)

#calculate both the probability and binary predictions
y_pred_proba = model.predict_proba(X_test)[:, 1]
y_pred = model.predict(X_test)

#create both the ROC AUC and accuracy scores
auc_score = roc_auc_score(y_test, y_pred_proba)
accuracy = accuracy_score(y_test, y_pred)

#print both the ROC AUC and accuracy scores
print(auc_score)
print(accuracy)

#create predictors for the model
X = df.drop(
columns=[
'target_7', 
'target_14', 
'Customer ID', 
'timestamp', 
'order_date'
]
)

#use the 2 week lag as the target variable
y = df['target_14']

#execute the train test split
X_train, X_test, y_train, y_test = train_test_split(
X, 
y, 
test_size=0.3, 
random_state=42
)

#create the model
model = RandomForestClassifier(
n_estimators = 500, 
max_depth = 10, 
min_samples_split = 5, 
random_state=42
)

#fit the model to the data
model.fit(X_train, y_train)

#calculate both the probability and binary predictions
y_pred_proba = model.predict_proba(X_test)[:, 1]
y_pred = model.predict(X_test)

#calculate both the ROC AUC and accuracy scores
auc_score = roc_auc_score(y_test, y_pred_proba)
accuracy = accuracy_score(y_test, y_pred)

#print both the ROC AUC and accuracy scores
print(auc_score)
print(accuracy)
