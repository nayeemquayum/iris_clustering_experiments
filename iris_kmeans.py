import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

#load data in bunch format
iris=load_iris()
#check the features of sklearn data set
print("iris features:",iris.feature_names)
#create dataframe from bunch
iris_df=pd.DataFrame(iris.data,columns=iris.feature_names)
print("iris features:",iris_df.info())
iris_df.rename(columns={'sepal length (cm)' : 'sepal_length_cm',
                     'sepal width (cm)' : 'sepal_width_cm',
                     'petal length (cm)': 'petal_length_cm',
                     'petal width (cm)' : 'petal_width_cm'}, inplace=True)
# #find enertia for different numbers of clusters
wcss = []
for i in range(1,11):
    km = KMeans(n_clusters=i)
    km.fit_predict(iris_df)
    wcss.append(km.inertia_)
#plot the graph of inertia vs cluster numbers
plt.plot(range(1,11),wcss)
plt.show()

#based on elbow technique we can see iris data could be classified in three groups
#run kmeans with cluster number 3
km_3 = KMeans(n_clusters=3)
y_prediction = km_3.fit_predict(iris_df)
print("y_prediction",y_prediction)
#let's separate the clusters
cluster_0=iris_df[y_prediction == 0]
cluster_1=iris_df[y_prediction == 1]
cluster_2=iris_df[y_prediction == 2]
print("cluster_0:",cluster_0.head())
#draw clusters on petal width and length
plt.scatter(cluster_0.petal_length_cm,cluster_0.petal_width_cm,color='blue')
plt.scatter(cluster_1.petal_length_cm,cluster_1.petal_width_cm,color='red')
plt.scatter(cluster_2.petal_length_cm,cluster_2.petal_width_cm,color='green')
plt.show()
