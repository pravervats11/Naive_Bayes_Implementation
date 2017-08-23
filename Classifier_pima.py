import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVR
import random
import operator

df = pd.read_csv("pima-indians.data.csv")

training_length = len(df)*0.67

training_data = df.ix[:int(training_length),:]
test_data = df.ix[int(training_length)+1:,:]

'''
Different Classifiers can be used. As all the classifiers have been imported.
Also Accuracy for this Dataset can be calculated for all the different classifiers.
'''


classifier = GaussianNB()
classifier.fit(training_data.ix[:,:-2], training_data.ix[:,-1])

predicted_labels = classifier.predict(test_data.ix[:,:-2])

expected_labels = test_data.ix[:,-1]

print(expected_labels,"\n\n", predicted_labels)

accuracy = classifier.score(test_data.ix[:,:-2], expected_labels)

print(accuracy)

