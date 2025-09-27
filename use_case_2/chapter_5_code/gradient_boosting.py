#assumes your data from the previous modeling steps are still in memory

#load in train test split
from sklearn.model_selection import train_test_split

#create the columns to predict the data
X = df.drop(columns=['target_7', 'target_14', 'Customer ID', 'timestamp', 'order_date'])

#using the 1 week lag as the target
y = df['target_7']

#execute the train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#load in packages for modeling
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score

#create the model
model = GradientBoostingClassifier(random_state=42)

#fit the model to the data
model.fit(X_train, y_train)

#create both probability and binary predictions
y_pred_proba = model.predict_proba(X_test)[:, 1]
y_pred = model.predict(X_test)

#calculate the ROC AUC and accuracy scores
auc_score = roc_auc_score(y_test, y_pred_proba)
accuracy = accuracy_score(y_test, y_pred)

#print the ROC AUC and accuracy scores
print(auc_score)
print(accuracy)

#create model with specific hyperparameters
model = GradientBoostingClassifier(learning_rate = .05, n_estimators = 500, subsample = 0.8, max_depth = 12, min_samples_split = 5, random_state=42)

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

#create the model with new hyperparameters
model = GradientBoostingClassifier(
learning_rate = .05, 
n_estimators = 500, 
subsample = 1, 
max_depth = 12, 
min_samples_split = 5, 
random_state=42
)

#fit the model to the data
model.fit(X_train, y_train)

#create both probability and binary predictions
y_pred_proba = model.predict_proba(X_test)[:, 1]
y_pred = model.predict(X_test)

#calculate both the ROC AUC and accuracy scores
auc_score = roc_auc_score(y_test, y_pred_proba)
accuracy = accuracy_score(y_test, y_pred)

#print both the ROC AUC and accuracy scores
print(auc_score)
print(accuracy)

#create model with updated hyperparameters
model = GradientBoostingClassifier(
learning_rate = .25, 
n_estimators = 500, 
subsample = 1, 
max_depth = 12, 
min_samples_split = 5, 
random_state=42
)

#fit model to the data
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

#load the grid search library
from sklearn.model_selection import GridSearchCV

#set up the hyperparameter grid
param_grid = {
    'n_estimators': [300, 500],
    'max_depth': [5, 10],
    'min_samples_split': [5, 10],
    'learning_rate': [0.07, 0.1, 0.2],
    'subsample': [0.8, 1]
}

#create the model
model = GradientBoostingClassifier(random_state=42)

#set the grid search
grid_search = GridSearchCV(estimator=model,
                           param_grid=param_grid,
                           cv=4, 
                           scoring='roc_auc',
                           n_jobs=-1,
                           verbose=1)

#execute the grid search
grid_search.fit(X_train, y_train)

#identify the best parameters for the model
best_model = grid_search.best_estimator_
print("Best Parameters:", grid_search.best_params_)


#set up predictors for the model
X = df.drop(
columns=[
'target_7', 
'target_14', 
'Customer ID', 
'timestamp', 
'order_date'
]
)

#set the target as the 2 week column
y = df['target_14']

#execute the train test split
X_train, X_test, y_train, y_test = train_test_split(
X, 
y, 
test_size=0.3, 
random_state=42
)

#create the model
model = GradientBoostingClassifier(
learning_rate = .25, 
n_estimators = 500, 
subsample = 1, 
max_depth = 12, 
min_samples_split = 5, 
random_state=42
)

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
