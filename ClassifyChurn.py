# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 15:45:29 2017
@author: Shubha Mishra
Algorithms applied : Neural Networks Multi-Layer Perceptron, SVM - Linear and RBF, Random Forest and Logistic Regression.
"""
import numpy as np
from sklearn.metrics import accuracy_score, mean_squared_error as mse
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

# Drop CustomerId column as it is not required
data.drop(['customerID'], axis = 1, inplace = True)

# Now let us work on categorical features. 
data.gender = [1 if x == "Male" else 0 for x in data.gender]
for col in ('Partner', 'Dependents', 'PhoneService' , 'OnlineSecurity',
        'OnlineBackup','DeviceProtection', 'TechSupport','StreamingTV',
        'StreamingMovies','PaperlessBilling','MultipleLines','Churn'):
    data[col] = [1 if x == "Yes" else 0 for x in data[col]]        
data.head(10)   # See how data looks like now
        
data.MultipleLines = pd.to_numeric(data.MultipleLines, errors = 'coerce')
data.TotalCharges = pd.to_numeric(data.TotalCharges, errors = 'coerce')

# Fill the missing values with 0
data['TotalCharges'] = data['TotalCharges'].fillna(0.0)

# Check for any existing missing values
print("Missing values now: \n", data.isnull().sum())

# Generate heatmap to visualize correlation between features to find least relevant features
print(data.corr()['Churn'].sort_values())
plt.figure(figsize = (12,12))
sns.heatmap(data.corr())         # Heatmap shows that Gender have very small (<0.01) correlation with Churn status

# For following features, let us generate bar plots w.r.t. target variable
for col in ('Partner', 'Dependents', 'PhoneService' , 'OnlineSecurity',
        'OnlineBackup','DeviceProtection', 'TechSupport','StreamingTV',
        'StreamingMovies','PaperlessBilling','MultipleLines'):
    sns.barplot(x = col, y = 'Churn', data = data)
    plt.show()
        
# Generate pairplots for high correlation features.
highCorrCols = ['MonthlyCharges','TotalCharges','tenure', 'Churn']
sns.pairplot(data[highCorrCols], hue = 'Churn')

data = pd.get_dummies(data = data)

# Prepare data for model training and testing input.
y = data.Churn.values     # Target feature

# All features except class (target)
X = data.drop(["Churn"],axis=1)

# Normalize data 
X = preprocessing.normalize(X, norm = "l2")

# Split the data into training and testing data
X_train, X_test, y_train, y_test  = train_test_split(X, y, test_size = 0.2, random_state=1)

# Classification using RBF SVM  
svc_rbf = SVC(kernel = "rbf")
svc_rbf = svc_rbf.fit(X_train,y_train)
prediction = svc_rbf.predict(X_test)
print("Mean-squared error using SVM RBF:", mse(y_test, prediction))
print("Accuracy with SVM RBF:",accuracy_score(y_test, prediction))

# Classification using Random Forest Classifier
rfc = RF(max_depth= 5, n_estimators= 10, max_features= 'auto')
rfc = rfc.fit(X_train,y_train)
prediction = rfc.predict(X_test)
print("Mean-squared error using Random Forest Classifier:", mse(y_test, prediction))
print("Accuracy with Random Forest Classifier:",accuracy_score(y_test, prediction))

# Classification using Logistic Regression
logreg = LR(C = 1)
logreg = logreg.fit(X_train,y_train)
prediction = logreg.predict(X_test)
print("Mean-squared error using Logistic Regression:", mse(y_test, prediction))
print("Accuracy with Logistic Regression:",accuracy_score(y_test, prediction))

# Classification using Multi-layer perceptron 
ann = MLPClassifier(solver='lbfgs', alpha = 1e-5,
                    hidden_layer_sizes = (5, 2), random_state = 1)
ann = ann.fit(X_train, y_train)
prediction = ann.predict(X_test)
print("Mean-squared error using Neural networks MLP:", mse(y_test, prediction))
print("Accuracy with Neural networks MLP:",accuracy_score(y_test, prediction))

# Since Multi-layer perceptron is senstitive to scaled features, let us standardize the data first and then generate model as well.
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
print("Mean-squared error using Neural networks MLP:", mse(y_test, prediction))
print("Accuracy with Neural networks MLP:",accuracy_score(y_test, prediction))

plt.show()

