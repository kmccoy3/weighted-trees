#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
Python script to run simulation.
"""


# =============================================================================
# AUTHOR INFORMATION
# =============================================================================


__author__ = "Kevin McCoy"
__copyright__ = "Copyright 2024, McCoy and Peterson"
__credits__ = ["Kevin McCoy", "Christine Peterson"]
__license__ = "MIT"
__version__ = "0.1.0"
__maintainer__ = "Christine Peterson"
__email__ = ["CBPeterson@mdanderson.org", "kmccoy1@rice.edu"]
__status__ = "development"
__date__ = "2024-10-08" # Last modified date


# =============================================================================
# IMPORTS
# =============================================================================


# Basic Imports
import numpy as np
from scipy.stats import uniform, invwishart, matrix_normal, norm
import pandas as pd
from sklearn.metrics import mean_squared_error as mse
import matplotlib.pyplot as plt
import seaborn as sns
from time import localtime, strftime

# sklearn imports
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.tree import DecisionTreeRegressor as DTR
from sklearn.linear_model import LogisticRegression as LR

# statsmodels imports
import statsmodels.formula.api as smf


# =============================================================================
# FUNCTIONS
# =============================================================================


def generate_data(seed=0, p=5, n=25, k=25, random_sigma=1, noise_sigma=1):

    # Set random seed
    np.random.seed(seed)

    # Generate main covariates
    M = np.random.uniform(-1, 1, size=(k, p))  # group means
    # M = np.zeros((k, p))

    U = invwishart.rvs(df=k+1, scale=np.identity(k))  # row covariance (group covariance)
    # U = np.identity(k)

    V = np.identity(p)

    X = matrix_normal.rvs(mean=M, rowcov=U, colcov=V, size=n)

    # Reshape X to be a 2D array
    X = X.reshape((n * k, p))

    # Generate group labels
    K = np.tile(np.arange(k), n)

    # Create dataframe
    df = pd.DataFrame(X, columns=["X" + str(i) for i in range(1, p + 1)])

    # Generate random effects and common noise terms
    noise = np.random.normal(0, noise_sigma, n * k)

    if p != 5:
        raise ValueError("Friedman function requires 5 covariates!")

    out = (
        np.sin(np.pi * df["X1"] * df["X2"])
        + 2 * (df["X3"] - 0.5) ** 2
        + df["X4"]
        + 0.5 * df["X5"]
    )

    # out = df["X1"] + df["X2"] + df["X3"] + df["X4"] + df["X5"]

    df["intercept"] = 1

    df["group"] = K

    df["random_effect"] = 0.0

    M = np.zeros((k, p+1))
    V = random_sigma * np.identity(p+1)

    random_effs = matrix_normal.rvs(mean=M, rowcov=U, colcov=V, size=1)


    for i in range(k):
        X = df[df["group"] == i].drop(columns=["group", "random_effect"])

        # print(X.columns)
        group_alpha = random_effs[i, :]
        random_effect = np.matmul(X, group_alpha)

        df.loc[df["group"] == i, "random_effect"] = random_effect

    df["noise"] = noise

    df["Y"] = out + noise + df["random_effect"]

    df["fake"] = norm.rvs(loc=0, scale=1, size=(n * k, 1))

    # Standardize Y
    df["Y"] = (df["Y"] - df["Y"].mean()) / df["Y"].std()

    # Output final dataframe
    return df


def main():

    # Initialize list to hold outputs
    results_list = []

    # Iterate over random sigmas
    for random_sigma in [0.5, 1, 2, 5]:

        # Iterate over seeds
        for seed in range(num_seeds):

            # Generate data
            data = generate_data(
                seed=seed,
                p=p,
                n=n,
                k=k,
                random_sigma=random_sigma,
                noise_sigma=noise_sigma
            )

            # Train-test split
            train_ratio = 0.8
            last_group = int(k * train_ratio)
            X_train = data[data["group"] < last_group]
            X_test = data[data["group"] >= last_group]

            ## Regular Tree
            tree = DTR()
            tree.fit(X_train[["X1", "X2", "X3", "X4", "X5", "fake"]], X_train["Y"])
            pred = tree.predict(X_test[["X1", "X2", "X3", "X4", "X5", "fake"]])
            tree_mse = mse(X_test["Y"], pred)
            results_list.append(
                {
                    "method": "Decision Tree",
                    "MSE": tree_mse,
                    "random_sigma": random_sigma,
                    "seed": seed,
                }
            )

            ## Regular random forest
            tree = RFR(n_estimators=last_group)
            tree.fit(X_train[["X1", "X2", "X3", "X4", "X5", "fake"]], X_train["Y"])
            pred = tree.predict(X_test[["X1", "X2", "X3", "X4", "X5", "fake"]])
            forest_mse = mse(X_test["Y"], pred)
            results_list.append(
                {
                    "method": "Random Forest",
                    "MSE": forest_mse,
                    "random_sigma": random_sigma,
                    "seed": seed,
                }
            )


            ## Regular Linear Mixed Model
            df = X_train.drop(columns=["noise", "random_effect", "intercept", "group"])
            md = smf.mixedlm("Y ~ X1 + X2 + X3 + X4 + X5 + fake", df, groups=X_train["group"], re_formula="~ X1 + X2 + X3 + X4 + X5 + fake")
            mdf = md.fit()
            pred = mdf.predict(X_test)
            lmm_mse = mse(X_test["Y"], pred)
            results_list.append(
                {
                    "method": "Linear Mixed Model",
                    "MSE": lmm_mse,
                    "random_sigma": random_sigma,
                    "seed": seed,
                }
            )


            ###############################################################################

            # Build group classifier
            group_clf = LR()  # GNB()  # DTC or RFC or LR or something else?
            group_clf.fit(
                X_train[["X1", "X2", "X3", "X4", "X5", "fake"]], X_train["group"]
            )
            group_pred = group_clf.predict_proba(
                X_test[["X1", "X2", "X3", "X4", "X5", "fake"]]
            )

            group_pred = group_pred # + 0.5 #

            # Normalize group predictions
            row_sums = group_pred.sum(axis=1)
            group_pred = group_pred / row_sums[:, np.newaxis]

            for test_group in range(last_group, k):
                rows = np.where(X_test["group"] == test_group)
                average = np.mean(group_pred[rows,], axis=1)
                group_pred[rows] = average

            ## Weighted Sum-of-Trees
            list_of_trees = []
            for i in range(last_group):
                tree = DTR()
                tree.fit(
                    X_train[X_train["group"] == i][
                        ["X1", "X2", "X3", "X4", "X5", "fake"]
                    ],
                    X_train[X_train["group"] == i]["Y"],
                )
                list_of_trees.append(tree)

            preds = np.array(
                [
                    tree.predict(X_test[["X1", "X2", "X3", "X4", "X5", "fake"]])
                    for tree in list_of_trees
                ]
            )
            preds = preds.T
            num = preds.shape[0]
            pred = [np.dot(preds[i, :], group_pred[i, :]) for i in range(num)]
            my_mse = mse(X_test["Y"], pred)
            results_list.append(
                {
                    "method": "Weighted Sum-of-Trees",
                    "MSE": my_mse,
                    "random_sigma": random_sigma,
                    "seed": seed,
                }
            )


        print("Finished Random Sigma: ", random_sigma)

    # Save results in csv and plot
    df = pd.DataFrame(results_list)
    curr_datetime = strftime(f"%Y-%m-%d_%H-%M", localtime())
    filename = "./out/" + curr_datetime + test_name
    df.to_csv(filename + ".csv", index=False)


    sns.boxplot(
        data=df,
        x="random_sigma",
        y="MSE",
        hue="method",
        palette="Paired",
        hue_order=[
            "Linear Mixed Model",
            "Decision Tree",
            "Random Forest",
            "Weighted Sum-of-Trees",
        ],
    )
    plt.rcParams['text.usetex'] = True
    plt.xlabel(r'$\sigma_\alpha$')
    plt.savefig(filename + ".png", dpi=600)

    # Save parameters
    with open(filename + ".txt", "a") as f:
        print("datetime:", curr_datetime, file=f)
        print("p: ", p, file=f)
        print("n: ", n, file=f)
        print("k: ", k, file=f)
        print("random_sigma: ", random_sigma, file=f)
        print("noise_sigma: ", noise_sigma, file=f)
        print("num_seeds: ", num_seeds, file=f)
        print("test_name: ", test_name, file=f)
        print("Group CLF:", group_clf, file=f)


# =============================================================================
# MAIN FUNCTION
# =============================================================================


if __name__ == "__main__":

    test_name = "lownhighk"

    # Simulated Data parameters
    p = 5
    n = 10
    k = 40
    noise_sigma = 1
    num_seeds = 20

    print("...starting...")

    main()

    print("Simulation complete.")
