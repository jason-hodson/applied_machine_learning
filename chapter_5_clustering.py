#import required libraries
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

#create sample data
n_samples = 300
random_state = 42
X, y = make_blobs(
n_samples=n_samples, 
random_state=random_state
)

#execute train test split
X_train, X_test = train_test_split(
X, 
test_size=0.3, 
random_state=random_state
)

#set number of clusters to 3
n_clusters = 3

#create kmeans clustering model
kmeans = KMeans(
n_clusters=n_clusters, 
random_state=random_state, 
n_init='auto'
)

#fit the model to the data
kmeans.fit(X)

#create the predictions on the test data
test_labels = kmeans.predict(X_test)

#create the predctions on the training data
train_labels = kmeans.predict(X_train)

#calculate and print the silhouette score for the training data
silhouette_avg = silhouette_score(X_train, train_labels)
print(silhouette_avg)

#calculate the silhouette score for the test data
silhouette_avg = silhouette_score(X_test, test_labels)
print(silhouette_avg)

#create predictions for full dataset
all_labels = kmeans.predict(X)

#plot the data in a scatterplot
plt.scatter(X[:, 0], X[:, 1], c=all_labels, s=50, cmap='viridis')

centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75)
plt.title('K-means Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()


#load required libraries
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

#create sample data
n_samples = 300
random_state = 42
X, y = make_blobs(n_samples=n_samples, random_state=random_state)

#execute train test split
X_train, X_test = train_test_split(
X, 
test_size=0.3, 
random_state=42
)

#create list of cluster options
cluster_options = [2, 3, 4, 5, 6]

#loop through the full clustering process
for n_clusters in cluster_options:
    kmeans = KMeans(
n_clusters=n_clusters, 
random_state=random_state, 
n_init='auto'
)
    kmeans.fit(X)
    test_labels = kmeans.predict(X_test)
    
    train_labels = kmeans.predict(X_train)
    silhouette_avg = silhouette_score(X_train, train_labels)
    print(n_clusters)
    print("Training: ", silhouette_avg)
    
    silhouette_avg = silhouette_score(X_test, test_labels)
    print("Test:", silhouette_avg)
    print("")

