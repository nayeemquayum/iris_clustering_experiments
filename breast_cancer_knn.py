import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

from sklearn.metrics import accuracy_score, recall_score,precision_score,f1_score, confusion_matrix

from ydata_profiling import ProfileReport
df = pd.read_csv('data/breast_cancer.csv')
print("breast cancer data:",df.info())
#prof = ProfileReport(df)
#prof.to_file(output_file='breast_cancer_report.html')
df.drop(['id','Unnamed: 32'],axis='columns',inplace=True)
print("breast cancer data:",df.info())
#split data in test and training set
X_train,X_test,y_train,y_test = train_test_split(df.drop(columns=['diagnosis']),
                                               df['diagnosis'],test_size=0.2,random_state=20)
#apply standard scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
knn = KNeighborsClassifier(n_neighbors=3)
#run cross validation for different numbers of clusters(k)
mean_accuracy_scores = []
for i in range(1, 15):
    print(f"Trying Knn with with {i} neighbors.")
    knn = KNeighborsClassifier(n_neighbors=i)
    mean_corss_validation_score = cross_val_score(knn, X_train, y_train, cv=10, scoring='accuracy').mean()
    print(f"mean 10 corss validation accuracy score for knn with {i} neighbor is:{mean_corss_validation_score}")
    mean_accuracy_scores.append(mean_corss_validation_score)
#plot the mean accuracy scores for k
plt.plot(range(1,15),mean_accuracy_scores)
plt.show()
#The graph shows for k=5 we get the best accuracy
print("Based on cross validation score, decided to set 5 neighbors for knn.")
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train,y_train)
y_prediction= knn.predict(X_test)
# Display Model Accuracy
print("Accuracy:",accuracy_score(y_test, y_prediction))
#confusion_matrix
confusion_matrix=pd.DataFrame(confusion_matrix(y_test,y_prediction),columns=list(range(0,2)))
print("Confusion matrix",confusion_matrix.head())
print("-"*25,"Classificaion Metrics","-"*25)
print("Precision:  ",precision_score(y_test,y_prediction,average='weighted'))
print("Recall: ",recall_score(y_test,y_prediction,average='weighted'))
print("F1 score: ",f1_score(y_test,y_prediction,average='weighted'))
print("-"*80)
#another way is using classification_report
from sklearn.metrics import classification_report
print(classification_report(y_test,y_prediction))

