#%% Load the dataset
get_ipython().magic(u'matplotlib inline')
import datetime
import time
import matplotlib.pyplot as plt
from sklearn import neighbors
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from operator import itemgetter
from itertools import groupby
import numpy as np
from LoadData import load

x,y=load()
# x=AggregativeBasicFeatures()
#%% Visualize the dataset for question a


#%% Test SMOTE for question b
from sklearn import linear_model
from sklearn import svm
from smote_testing import smote_testing_for_classifier
from sklearn import preprocessing
from sklearn import tree


classifier1 = svm.SVC()
classifier2 = linear_model.LogisticRegression(C=1e5)
classifier3 = tree.DecisionTreeRegressor()
x_normalized= preprocessing.scale(x)

#smote_testing_for_classifier(x_normalized,y,classifier1,'SVM')
smote_testing_for_classifier(x_normalized,y,classifier2,'Logistic regression')
#smote_testing_for_classifier(x_normalized,y,classifier3,'Three regressor')