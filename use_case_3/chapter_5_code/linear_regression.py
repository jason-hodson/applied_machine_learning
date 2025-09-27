####Linear regression

#read in pandas
import pandas as pd

#read in data from previous chapter
df = pd.read_csv("df_summarized")

#read in sklearn libraries for the modeling process
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error


#split data into what will be used to predict and what is being predicted
X = df.drop(columns=['DATE', 'AREA NAME', 'crime_count'])
y = df['crime_count']

#perform standard train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#create and fit the regression model
model = LinearRegression()
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

#extract the coefficents from the model
coefficients = model.coef_

#get the feature names from the model
feature_names = model.feature_names_in

#add the these into a series to display the coefficients alongside their feature names
pd.Series(data=coefficients, index=feature_names)

from sklearn.decomposition import PCA

#create the pca object using 5 components
pca = PCA(n_components=5)

#fit the pca model to the training data
pca.fit(X_train)

#apply the pca model to the X_train data
X_train = pca.transform(X_train)

#apply the pca model to the X_test data
X_test = pca.transform(X_test)

#check the explained variance ratio
print("Overall Explained Variance: ",sum(pca.explained_variance_ratio_))
print("Explained Variance Ratio: ", pca.explained_variance_ratio_)

#create and fit the regression model
model = LinearRegression()
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

