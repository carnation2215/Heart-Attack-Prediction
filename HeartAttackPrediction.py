import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
#from numpy import nan


import time

import warnings
warnings.filterwarnings(action="ignore")

pd.set_option('display.width', 1000)
pd.set_option('display.max_column', 25)


# Exploratory Analysis
# Load the dataset and do some quick exploratory analysis.

data = pd.read_csv('HeartAttack_data.csv', index_col=False)
print("\n\n\nSample HeartAttack dataset :- \n\n", data.head(5) )


print("\n\n\nShape of the HeartAttack dataset :- \n")
print( data.shape)


print("\n\n\n HeartAttack data decription :- \n")
print( data.describe(include="all") )


list_drop = ['slope',	'ca'	,'thal']
data.drop(list_drop, axis=1, inplace=True)
print("\nAfter Deletion of column :-\n", data)

data.replace('?',np.nan,inplace=True)
print("\nAfter replacing the ? symbol by integer value. Update columns are :-\n", data)

print(data.isnull().sum())
column=['trestbps','chol','fbs','restecg','thalach','exang']
data[column]=data[column].fillna(data.mode().iloc[0])
data['age'].fillna(data['age'].mean(),inplace=True)
data['sex'].fillna(data['sex'].mean(),inplace=True)
data['cp'].fillna(data['cp'].mean(),inplace=True)
data['oldpeak'].fillna(data['oldpeak'].mean(),inplace=True)
print(data)


data.loc[:,'trestbps':'exang']=data.loc[:,'trestbps':'exang'].applymap(float)
data.astype('float64')
print(data.dtypes)
print(data.info())

print( "\ndata.num.unique() : " , data.num.unique() )

print("\ndata.groupby('num').size()\n")
print(data.groupby('num').size())

#num
# 0    188
# 1    106
# dtype: int64


plt.hist(data['num'])
plt.title('Diagnosis of Heart Attack (Male=1 , Female=0)')
plt.show()


# Next, we visualise the data using density plots to get a sense of the data distribution.

data.plot(kind='density', subplots=True, layout=(4,3), sharex=False, legend=True, fontsize=2)
plt.show()

# from matplotlib import cm as cm

fig = plt.figure()
ax1 = fig.add_subplot(111)
cax = ax1.imshow(data.corr() )
ax1.grid(True)
plt.title('HeartAttack Attributes Correlation')
# Add colorbar
fig.colorbar(cax, ticks=[.75,.8,.85,.90,.95,1])
plt.show()

# Finally, we'll split the data into predictor variables and target variable,
# following by breaking them into train and test sets.


Y = data['num'].values
X = data.drop('num', axis=1).values

X_train, X_test, Y_train, Y_test = train_test_split (X, Y, test_size = 0.33, random_state=21)

models_list = []
models_list.append(('CART', DecisionTreeClassifier()))
models_list.append(('SVM', SVC()))
models_list.append(('NB', GaussianNB()))
models_list.append(('KNN', KNeighborsClassifier()))
models_list.append(('LDA', LinearDiscriminantAnalysis()))
num_folds = 10

results = []
names = []

print("\nAccuracies of algorithm before scaled dataset :-\n")

for name, model in models_list:
    kfold = KFold(n_splits=num_folds, random_state=123)
    startTime = time.time()
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
    endTime = time.time()
    results.append(cv_results)
    names.append(name)
    print( "%s :-  %f (%f) (run time: %f)" % (name, cv_results.mean(), cv_results.std(), endTime-startTime))

#CART :-  0.725789 (0.074412) (run time: 0.034814)
#SVM :-  0.662105 (0.150889) (run time: 0.040696)
#NB :-  0.821579 (0.087357) (run time: 0.015637)
#KNN :-  0.611316 (0.124716) (run time: 0.046871)
#LDA :-  0.805789 (0.090404) (run time: 0.026142)


#Performance Comparision
#------------------------------


fig = plt.figure()
fig.suptitle('Performance Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

# Standardize the dataset


pipelines = []

pipelines.append(('ScaledCART', Pipeline([('Scaler', StandardScaler()),('CART', DecisionTreeClassifier())])))
pipelines.append(('ScaledSVM', Pipeline([('Scaler', StandardScaler()),('SVM', SVC( ))])))
pipelines.append(('ScaledNB', Pipeline([('Scaler', StandardScaler()),('NB', GaussianNB())])))
pipelines.append(('ScaledKNN', Pipeline([('Scaler', StandardScaler()),('KNN', KNeighborsClassifier())])))
pipelines.append(('ScaledLDA', Pipeline([('Scaler', StandardScaler()),('LDA', LinearDiscriminantAnalysis())])))

results = []
names = []


print("\nAccuracies of algorithm after scaled dataset :-\n")

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    kfold = KFold(n_splits=num_folds, random_state=123)
    for name, model in pipelines:
        start = time.time()
        cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
        end = time.time()
        results.append(cv_results)
        names.append(name)
        print( "%s :-  %f (%f) (run time: %f)" % (name, cv_results.mean(), cv_results.std(), end-start))

#ScaledCART :-  0.700263 (0.071948) (run time: 0.053448)
#ScaledSVM :-  0.801053 (0.125282) (run time: 0.069008)
#ScaledNB :-  0.821579 (0.087357) (run time: 0.046882)
#ScaledKNN :-  0.806316 (0.105389) (run time: 0.069020)
#ScaledLDA :-  0.805789 (0.090404) (run time: 0.062495)

#Performance Comparison after Scaled Data
#----------------------------------------

fig = plt.figure()
fig.suptitle('Performance Comparison after Scaled Data')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

# Application on GaussianNB Dataset

# Prepare the model

scaler = StandardScaler().fit(X_train)

X_train_scaled = scaler.transform(X_train)
model =GaussianNB()
start = time.time()
model.fit(X_train_scaled, Y_train)   #Training of algorithm
end = time.time()
print( "\nGaussianNB Training Completed. It's Run Time is :-  %f \n" % (end-start))

#Run Time: 0.015625

# Estimate accuracy on test dataset

X_test_scaled = scaler.transform(X_test)
predictions = model.predict(X_test_scaled)
print(" All predictions done successfully by GaussianNB Machine Learning Algorithms\n ")
print("\nAccuracy score :-  %f" % accuracy_score(Y_test, predictions))

# Accuracy score 0.836735 = 84% (Approx)

print("\n\n")
print("Confusion Matrix = \n")
print( confusion_matrix(Y_test, predictions))

from sklearn.externals import joblib
filename =  "finalized_HeartAttack_model.sav"
joblib.dump(model, filename)
print("\nBest Performing Model dumped successfully into a file by Joblib. \n ")

print("-*-"  * 54 )

print(" MACHINE LEARNING SOLUTION proposed by : Divya Mishra ")
print(" Email ID : divyamishra1505@gmail.com ")
print(" Student Order Id : 381015 ")




