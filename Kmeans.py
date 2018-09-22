# -*- coding: utf-8 -*-
"""
Created on Thur Sept 20 15:45:29 2018
@author: Shubha Mishra
Application of K-means clustering on Teleco Customer Churn dataset.
"""
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing 
from sklearn.cluster import KMeans

filename = "./Telco-Customer-Churn.csv"

# Read the data
data = pd.read_csv(filename)
# Overview and statistical details of the data..
data.head()
data.info()
data.describe()

# Check if there are any missing values.
print("Missing values:",data.isnull().sum())

data.drop(['customerID'], axis = 1, inplace = True)
data.gender = [1 if x == "Male" else 0 for x in data.gender]
data.Churn = [1 if x == "Yes" else 0 for x in data.Churn ]
data.TotalCharges = pd.to_numeric(data.TotalCharges, errors = 'coerce')

# See correlation between other features and Churn to find irrelevant features
data.corr()['Churn'].sort_values()

# Remove irrelevant (low correlation) features
data.drop(['SeniorCitizen','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','PaymentMethod'],axis=1,inplace=True)

# Delete rows with missing/null values
data.dropna(inplace = True)
data = pd.get_dummies(data = data)

# Prepare data for model training and testing input.
y = data.Churn.values     # Target feature

# All features except class (target)
X = data.drop(["Churn"],axis=1)

# Normalize data 
X = preprocessing.normalize(X, norm = "l2")

#Assign number of clusters required. Here, 2 since only two classes.
n_clusters = 2

kmeans = KMeans(n_clusters = n_clusters, random_state = None, n_init = 10,
                tol = 0.001, max_iter = 5000, init = 'k-means++')
kmeans.fit(input_file, y)
predicted = (kmeans.fit_predict(input_file))

# Compute confusion matrix and accuracy.
print(confusion_matrix(y, predicted))
print(" ")
print(accuracy_score(y, predicted))

# Plot to see clusters and their centers.
plt.scatter(input_file[:, 0], input_file[:, 1], c = predicted, s = 20)
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c ='red', s= 30, alpha=0.5);
plt.show()
