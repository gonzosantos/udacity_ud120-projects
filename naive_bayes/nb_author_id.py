#!/usr/bin/python3

""" 
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project. 

    Use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()


##############################################################
# Enter Your Code Here
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
clf = GaussianNB()

#print(features_train[:10])
print('Train Features length: ' + str(len(features_train)))
#print(labels_train[:10])
print('Train Labels length: ' + str(len(labels_train)))
#print(features_test[:10])
print('Test Features length: ' + str(len(features_test)))
#print(labels_test[:10])
print('Test Labels length: ' + str(len(labels_test)))

t0 = time()
clf.fit(features_train, labels_train)
print("Training Time:", round(time()-t0, 3), "s")

t1 = time()
labels_pred = clf.predict(features_test)
#print(labels_pred.reshape(len(labels_pred),1))
#print(labels_test.reshape(len(labels_test),1))
print(accuracy_score(labels_test, labels_pred))
print("Predicting Time:", round(time()-t1, 3), "s")
##############################################################

##############################################################
'''
You Will be Required to record time for Training and Predicting 
The Code Given on Udacity Website is in Python-2
The Following Code is Python-3 version of the same code
'''

# t0 = time()
# # < your clf.fit() line of code >
# print("Training Time:", round(time()-t0, 3), "s")

# t0 = time()
# # < your clf.predict() line of code >
# print("Predicting Time:", round(time()-t0, 3), "s")

##############################################################