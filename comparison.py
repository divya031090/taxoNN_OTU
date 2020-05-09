from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn import linear_model
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
import math
from datetime import datetime

from data_read import load

# Load data
X, Y, P, Q = load()
lb = LabelEncoder()
#1 hot-encoding
Y_new=y = lb.fit_transform(Y)
Q_new=y = lb.fit_transform(Q)




clf = RandomForestClassifier(
	n_estimators=500,random_state=0).fit(X, Y.values.ravel())
print ("Accuracy of Random Forest Classifier: "+str(clf.score(P,Q)))

clf2 = SVC(kernel='rbf',C=10,
	gamma=0.001,random_state=0).fit(X, Y.values.ravel())
print ("Accuracy of SVM: "+str(clf2.score(P,Q)))

clf3 = BernoulliNB().fit(X, Y.values.ravel())
print ("Accuracy of Naive Bayes Classifier: "+str(clf3.score(P,Q)))



clf4 = GaussianNB().fit(X, Y.values.ravel())
print ("Accuracy of Gaussian  Bayes Classifier: "+str(clf4.score(P,Q)))

clf5 = linear_model.LinearRegression()
clf5.fit(X, Y_new.ravel())
print ("Accuracy of Linear Regression: "+str(clf5.score(P,Q_new)))


clf6 = linear_model.Ridge()
clf6.fit(X, Y_new.ravel())
print ("Accuracy of Ridge Regression: "+str(clf4.score(P,Q)))

clf7 = linear_model.Lasso()
clf7.fit(X, Y_new.ravel())
print ("Accuracy of Lasso Regression: "+str(clf4.score(P,Q)))









