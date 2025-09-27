#Assumes your data are still in memory from the logistic regression section

#read in packages to prep data and calculate accuracy
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score

#drop columns to define X
X = df.drop(columns=['target_7', 'target_14', 'Customer ID', 'timestamp', 'order_date'])

#define y as the one week lag
y = df['target_7']

#split into train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#read in decision tree classifier library
from sklearn.tree import DecisionTreeClassifier

#create model
dt_classifier = DecisionTreeClassifier(random_state=42)

#fit the model to the data
dt_classifier.fit(X_train, y_train)

#create both the probability and binary predictions
y_pred_proba = dt_classifier.predict_proba(X_test)[:, 1]
y_pred = dt_classifier.predict(X_test)

#calculate both the ROC AUC and accuracy scores
auc_score = roc_auc_score(y_test, y_pred_proba)
accuracy = accuracy_score(y_test, y_pred)

#print both the ROC AUC and accuracy scores
print(accuracy)
print(auc_score)

#create the model with a max_depth of 4
dt_classifier = DecisionTreeClassifier(max_depth = 4, random_state=42)

#fit the model
dt_classifier.fit(X_train, y_train)

#calculate the binary and probability prediction
y_pred_proba = dt_classifier.predict_proba(X_test)[:, 1]
y_pred = dt_classifier.predict(X_test)

#calculate the ROC AUC and accuracy scores
auc_score = roc_auc_score(y_test, y_pred_proba)
accuracy = accuracy_score(y_test, y_pred)

#print the ROC AUC and accuracy scores
print(accuracy)
print(auc_score)

#import plotting libraries
import matplotlib.pyplot as plt
from sklearn import tree

#plot the decision tree
plt.figure(figsize=(20, 15))
_ = tree.plot_tree(dt_classifier,
                   feature_names=list(X_train.columns),
                   class_names=['No', 'Yes'],
                   filled=True,
                   rounded=True,
                   fontsize=10)
plt.show()


#create the model with a max_depth of 3
dt_classifier = DecisionTreeClassifier(max_depth = 3, random_state=42)

#fit the model to the data
dt_classifier.fit(X_train, y_train)

#calculate the binary and probability predictions
y_pred_proba = dt_classifier.predict_proba(X_test)[:, 1]
y_pred = dt_classifier.predict(X_test)

#calculate the ROC AUC and accuracy scores
auc_score = roc_auc_score(y_test, y_pred_proba)
accuracy = accuracy_score(y_test, y_pred)

#print the ROC AUC and accuracy scores
print(accuracy)
print(auc_score)

#create the dataset for predictors for the model
X = df.drop(
columns=['target_7', 'target_14', 'Customer ID', 'timestamp', 'order_date'])

#create y as the 2 week target
y = df['target_14']

#execute the train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#create model with max_depth of 3
dt_classifier = DecisionTreeClassifier(max_depth = 3, random_state=42)

#fit the model
dt_classifier.fit(X_train, y_train)

#calculate both the probability and binary predictions
y_pred_proba = dt_classifier.predict_proba(X_test)[:, 1]
y_pred = dt_classifier.predict(X_test)

#calculate the ROC AUC and accuracy scores
auc_score = roc_auc_score(y_test, y_pred_proba)
accuracy = accuracy_score(y_test, y_pred)

#print both the ROC AUC and accuracy scores
print(accuracy)
print(auc_score)

#plot the decision tree
plt.figure(figsize=(20, 15))
_ = tree.plot_tree(dt_classifier,
                   feature_names=list(X_train.columns),
                   class_names=['No', 'Yes'],
                   filled=True,
                   rounded=True,
                   fontsize=10)
plt.show()


