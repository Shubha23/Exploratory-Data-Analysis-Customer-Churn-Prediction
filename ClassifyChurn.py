# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 15:45:29 2017
@author: Shubha Mishra
Algorithms applied : Neural Networks Multi-Layer Perceptron, SVM - Linear and RBF, Random Forest and Logistic Regression.
"""
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.linear_model import LogisticRegression as LR
from sklearn.svm import SVC 
from sklearn.neural_network import MLPClassifier

filename = "./Telco-Customer-Churn.csv"

# Read the data
data = pd.read_csv(filename)
# Overview and statistical details of the data..
data.head()
data.info()
data.describe()

sizes = data['Churn'].value_counts(sort = False)
labels = np.unique(data.Churn)

# Visualize the data
plt.figure(figsize = (8,8))
plt.subplot(212)
#data['Churn'].value_counts().plot.pie()
plt.title("Customer churn rate:")
plt.pie(sizes, labels = labels, autopct='%1.1f%%')   # Distribution shows about 26% customers churn

# Find if there are any missing values.
print("Missing values:",data.isnull().sum())     # No missing data is found, so nothing to do

data.drop(['customerID'], axis = 1, inplace = True)
# Create dummy variables for categorical features
data.gender = [1 if x == "Male" else 0 for x in data.gender]
for col in ('Partner', 'Dependents', 'PhoneService' , 'OnlineSecurity',
        'OnlineBackup','DeviceProtection', 'TechSupport',
        'StreamingTV','StreamingMovies','PaperlessBilling',
        'MultipleLines','Churn'):
    data[col] = [1 if x == "Yes" else 0 for x in data[col]]
        
data.MultipleLines = pd.to_numeric(data.MultipleLines, errors = 'coerce')
data.TotalCharges = pd.to_numeric(data.TotalCharges, errors = 'coerce')

# Generate heatmap to visualize correlation between features to find least relevant features
print(data.corr()['Churn'].sort_values())
plt.figure(figsize = (12,12))
sns.heatmap(data.corr())         # Heatmap shows that Gender have very small (<0.01) correlation with Churn status

# Remove features with correlation coefficient between +/-0.05 (considering it as threshold)
data.drop(['gender','PhoneService','MultipleLines','PaymentMethod'],axis=1,inplace=True)

data = pd.get_dummies(data = data)

# Prepare data for model training and testing input.
y = data.Churn.values     # Target feature

# All features except class (target)
X = data.drop(["Churn"],axis=1)

# Normalize data 
X = preprocessing.normalize(X, norm = "l2")

# Split the data into training and testing data
X_train, X_test, y_train, y_test  = train_test_split(X, y, test_size = 0.2, random_state=1)

# Classification using Linear SVM
svc_l = SVC(kernel="linear", C = 0.025)
svc_l = svc_l.fit(X_train,y_train)
prediction = svc_l.predict(X_test)
#print("3. Linear SVM ", prediction)
print("Accuracy with Linear SVM:", accuracy_score(y_test, prediction))

# Classification using RBF SVM  
svc_rbf = SVC(kernel = "rbf", gamma= 1, C = 1)
svc_rbf = svc_rbf.fit(X_train,y_train)
prediction = svc_rbf.predict(X_test)
print("Accuracy with RBF SVM:",accuracy_score(y_test, prediction))

# Classification using Random Forest Classifier
rfc = RF(max_depth= 5, n_estimators = 10, max_features= 'auto')
rfc = rfc.fit(X_train,y_train)
prediction = rfc.predict(X_test)
print("Accuracy with Random Forest Classifier:",accuracy_score(y_test, prediction))

# Classification using logistic regression
logreg = LR(C = 1)
logreg = logreg.fit(X_train,y_train)
prediction = logreg.predict(X_test)
print("Accuracy with Logistic Regression:",accuracy_score(y_test, prediction))

# Since Multi-layer perceptron is senstitive to scaled features, standardize the data first
from sklearn.preprocessing import StandardScaler  
scaler = StandardScaler()  
# Fit only on training data
scaler.fit(X_train)  
X_train = scaler.transform(X_train)  
# Apply same transformation to test data
X_test = scaler.transform(X_test)

# Classification using Multi-layer perception 
ann = MLPClassifier(solver='lbfgs', alpha = 1e-5,
                    hidden_layer_sizes = (5, 2), random_state = 1)

ann = ann.fit(X_train, y_train)
prediction = ann.predict(X_test)
print("F1-score using Neural networks MLP:", f1_score(y_test, prediction))
print("Accuracy with Neural networks MLP:",accuracy_score(y_test, prediction))

plt.show()

