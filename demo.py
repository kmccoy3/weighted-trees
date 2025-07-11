#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
Python script containing a demo of the Weighted Sum-of-Trees method.
"""


# =============================================================================
# AUTHOR INFORMATION
# =============================================================================


__author__ = "Kevin McCoy"
__copyright__ = "Copyright 2025, McCoy et al."
__credits__ = ["Kevin McCoy", "Zachary Wooten", "Katarzyna Tomczak", "Christine Peterson"]
__license__ = "MIT"
__version__ = "0.1.0"
__maintainer__ = "Kevin McCoy"
__email__ = ["CBPeterson@mdanderson.org", "kmm12@rice.edu"]
__status__ = "release"
__date__ = "2025-07-11" # Last modified date


# =============================================================================
# IMPORTS
# =============================================================================


# Basic Imports
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error as mse
from time import time

# sklearn imports
from sklearn.tree import DecisionTreeRegressor as DTR
from sklearn.linear_model import LogisticRegression as LR


# =============================================================================
# MAIN FUNCTION
# =============================================================================


def main():

    # Generate data
    n = 100 # number of observations per group
    p = 5 # number of features
    k = 10 # number of groups
    data = np.random.rand(n * k, p) # ~ [0, 1] uniform distribution
    data = pd.DataFrame(data, columns=[f"X{i+1}" for i in range(p)]) # convert to DataFrame
    data["Y"] = data.sum(axis=1) + np.random.normal(0, 0.1, size=n * k) # add response variable
    data["group"] = np.tile(np.arange(k), n) # add group variable

    # Split data into training and test sets
    train_ratio = 0.8
    last_group = int(k * train_ratio)
    X_train = data[data["group"] < last_group]
    X_test = data[data["group"] >= last_group]
    input_vars = ["X1", "X2", "X3", "X4", "X5"]

    # Start timer
    curr_time = time()

    # Build group classifier
    group_clf = LR()
    group_clf.fit(X_train[input_vars], X_train["group"])
    group_pred = group_clf.predict_proba(X_test[input_vars])

    # Average class probabilities for each group
    for test_group in range(last_group, k):
        rows = np.where(X_test["group"] == test_group)
        average = np.mean(group_pred[rows,], axis=1)
        group_pred[rows] = average

    ## Build Sum-of-Trees
    list_of_trees = []
    for i in range(last_group):
        tree = DTR()
        tree.fit(X_train[X_train["group"] == i][input_vars], X_train[X_train["group"] == i]["Y"])
        list_of_trees.append(tree)

    # Predict using the Sum-of-Trees method
    preds = np.array([tree.predict(X_test[input_vars]) for tree in list_of_trees]) # make predictions
    preds = preds.T # shape (num_test, num_training_groups)
    num = preds.shape[0] # number of test observations
    pred = [np.dot(preds[i, :], group_pred[i, :]) for i in range(num)] # weighted sum
    my_mse = mse(X_test["Y"], pred) # calculate mean squared error

    # Print results
    print(f"Mean Squared Error: {my_mse:.4f}")
    print(f"Total Time Taken: {time() - curr_time:.4f} seconds")


# =============================================================================
# RUN MAIN FUNCTION
# =============================================================================


if __name__ == "__main__":
    main()
