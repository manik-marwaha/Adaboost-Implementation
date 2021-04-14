#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")


# In[2]:


# Read data with pandas
data = pd.read_csv(r"C:\Users\MANIK MARWAHA\Desktop\wdbc_data.csv",encoding="UTF-8")

# Add names to columns
data.columns = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12',
                '13', '14', '15', '16','17', '18', '19', '20', '21', '22',
                '23', '24', '25', '26', '27', '28', '29', '30', '31', '32']

# Separate y_train and y_test
y_train = data.iloc[:300, 1:2]
y_test = data.iloc[300:, 1:2]


# drop index column and labels
data.drop(columns=['1', '2'], axis=1, inplace=True)

# Separate X_train and X_test
X_train = data.iloc[:300]
X_test = data.iloc[300:]


# In[3]:


# Change y from 'M' / 'B' to -1 / 1
for i in range(len(y_train)):
    if y_train.loc[i].values == 'B':
        y_train.loc[i] = -1
    else:
        y_train.loc[i] = 1

for i in range(len(y_test )):
    i = i + 300
    if y_test.loc[i].values == 'B':
        y_test.loc[i] = -1
    else:
        y_test.loc[i] = 1

# Change y to desired format
y_train = y_train.astype('int')
y_test = y_test.astype('int')
y_train = np.reshape(y_train.values, 300)
y_test = np.reshape(y_test.values, 268)


# In[4]:


# GET ERROR RATE
def get_error_rate(pred, Y):
    return sum(pred != Y) / float(len(Y))

# FUNCTION TO PRINT ERROR RATE
def print_error_rate(err):
    print
    'Error rate: Training: %.4f - Test: %.4f' % err


# In[5]:


# GENERIC CLASSIFIER WHICH WILL BE USED AS A HELPER FUNCTION
def generic_clf(Y_train, X_train, Y_test, X_test, clf):
    clf.fit(X_train, Y_train)
    pred_train = clf.predict(X_train)
    pred_test = clf.predict(X_test)
    return get_error_rate(pred_train, Y_train),            get_error_rate(pred_test, Y_test)

# ADABOOST ALGORITHM IMPLEMENTATION
def adaboost_model(Y_train, X_train, Y_test, X_test, M, clf):
    number_train, number_test = len(X_train), len(X_test)

    # Initialize weights
    weight = np.ones(number_train) / number_train
    pred_train, pred_test = [np.zeros(number_train), np.zeros(number_test)]

    for i in range(M):

        # Fit a classifier with the specific weights
        clf.fit(X_train, Y_train, sample_weight=weight)
        pred_train_i = clf.predict(X_train)
        pred_test_i = clf.predict(X_test)

        # Indicator function
        ind = [int(x) for x in (pred_train_i != Y_train)]

        # used to update weights
        ind2 = [x if x == 1 else -1 for x in ind]

        # Error for mth iteration
        err_m = np.dot(weight, ind) / sum(weight)

        # Alpha - weights for mth classifier
        alpha_m = 0.5 * np.log((1 - err_m) / float(err_m))

        # New weights to be updated
        weight = np.multiply(weight, np.exp([float(x) * alpha_m for x in ind2]))

        # Add to prediction
        pred_train = [sum(x) for x in zip(pred_train,
                                          [x * alpha_m for x in pred_train_i])]
        pred_test = [sum(x) for x in zip(pred_test,
                                         [x * alpha_m for x in pred_test_i])]

    pred_train, pred_test = np.sign(pred_train), np.sign(pred_test)

    # Return error rate in train and test set
    return get_error_rate(pred_train, Y_train),            get_error_rate(pred_test, Y_test)


# In[6]:


# Fit a simple decision tree first as base classifier (DECISION STUMPS)

clf_tree = DecisionTreeClassifier(max_depth=1, random_state=1) # Decision Stumps

# calculate error using generic classifier
error_tree = generic_clf(y_train, X_train, y_test, X_test, clf_tree)
error_train, error_test = [error_tree[0]], [error_tree[1]]
error_i = adaboost_model(y_train, X_train, y_test, X_test, 50, clf_tree)
print(1-error_i[1])


# In[7]:


# PLOT FUNCTION FOR ERROR VS NUMBER OF ITERTIONS

def plot_error_rate(er_train, er_test):
    df_error = pd.DataFrame([er_train, er_test]).T
    df_error.columns = ['Training', 'Test']
    plot1 = df_error.plot(linewidth=3, figsize=(8, 6),
                          color=['lightgreen', 'darkgreen'], grid=True)
    plot1.set_xlabel('N_iterations', fontsize=12)
    plot1.set_xticklabels(range(0, 450, 50))
    plot1.set_ylabel('Error rate', fontsize=12)
    plot1.set_title('Error rate vs number of iterations', fontsize=16)
    plt.axhline(y=er_test[0], linewidth=1, color='blue', ls='dashed')
    plt.show()


# In[8]:


## CHECKING WITH DIFFERENT ITERATIONS TO PLOT

# Fit a simple decision tree first (DECISION STUMPS)

clf_tree = DecisionTreeClassifier(max_depth=1, random_state=1) # Decision Stumps
error_tree = generic_clf(y_train, X_train, y_test, X_test, clf_tree)

# Fit Adaboost classifier using a decision tree as base estimator
# Test with different number of iterations
error_train, error_test = [error_tree[0]], [error_tree[1]]
x_range = range(10, 410, 10)
for i in x_range:
    error_i = adaboost_model(y_train, X_train, y_test, X_test, i, clf_tree)
    error_train.append(error_i[0])
    error_test.append(error_i[1])

# Compare error rate vs number of iterations
plot_error_rate(error_train, error_test)


# In[11]:


# using inbuilt sklearn package to compare results
from sklearn import metrics
clf = AdaBoostClassifier(n_estimators=50, random_state=0)
clf.fit(X_train, y_train)
pred = clf.predict(X_test)
print(f'accuracy: {metrics.accuracy_score(y_test, pred)}')


# In[10]:


# comparing results with inbuilt SVM
from sklearn import metrics
from sklearn.svm import SVC

clf = SVC(C = 1, kernel = 'linear')

clf.fit(X_train, y_train)
pred2 = clf.predict(X_test)
print(f'accuracy: {metrics.accuracy_score(y_test, pred2)}')

