# Basic Imports
import numpy as np
from scipy.stats import uniform, invwishart, matrix_normal, norm
from scipy.stats import multivariate_normal as mvn
import pandas as pd
from sklearn.metrics import mean_squared_error as mse
import matplotlib.pyplot as plt
import seaborn as sns
from time import localtime, strftime, time

# sklearn imports
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.tree import DecisionTreeRegressor as DTR
from sklearn.linear_model import LogisticRegression as LR
from sklearn.naive_bayes import GaussianNB as GNB

# statsmodels imports
import statsmodels.api as sm
import statsmodels.formula.api as smf


# Define basis functions
def f_0(x):
    return 10*np.sin(np.pi * x[:,0] * x[:,1])

def f_1(x):
    return 10*(x[:,2] - 0.5)**2

def f_2(x):
    return 10*(x[:,2] - 0.5)**2 + 10*x[:,3] + 5*x[:,4]

def f_3(x):
    return 6*x[:,0] + (4-10*(x[:,1] > 0.5)) * np.sin(np.pi * x[:,0]) - 4*(x[:,1] > 0.5) + 15

# Define 3 Unique Data Generating Processes (DGPs)
def mu_1(x, groups):
    return (f_0(x) + f_1(x) + f_2(x) - 0.75) * (groups % 2 == 0) + f_3(x) * (groups % 2 == 1)

def mu_2(x, groups):
    return f_0(x) * (groups % 3 == 0) + f_1(x) * (groups % 3 == 1) + f_2(x) * (groups % 3 == 2) + f_3(x) * (groups % 2 == 0)

def mu_3(x, groups):
    return f_0(x) * np.logical_or(groups % 3 == 0, groups % 3 == 1) + f_1(x) * np.logical_or(groups % 3 == 1, groups % 3 == 2) + f_2(x) * np.logical_or(groups % 3 == 2, groups % 3 == 0) + f_3(x) * (groups % 2 == 0)



def generate_data(seed=0, p=5, n=25, k=25, random_sigma=1, noise_sigma=1, mu=1):

    # Set random seed
    np.random.seed(seed)

    # Generate main covariates
    M = np.random.uniform(0, 2, size=(k, p))  # group means
    U = np.identity(k)
    V = np.identity(p)
    X = matrix_normal.rvs(mean=M, rowcov=U, colcov=V, size=n)

    # Reshape X to be a 2D array
    X = X.reshape((n * k, p))

    # Generate group labels
    K = np.tile(np.arange(k), n) # groups 0, 1, 2, ..., k-1, 0, 1, ....

    # Create dataframe
    df = pd.DataFrame(X, columns=["X" + str(i) for i in range(1, p + 1)])

    # Generate random effects and common noise terms
    noise = np.random.normal(0, noise_sigma, n * k)

    # Add group labels to the dataframe
    df["group"] = K

    # Generate Y according to the specified Data Generating Process (DGP)
    if mu == 1:
        out = mu_1(X, K)
    elif mu == 2:
        out = mu_2(X, K) 
    elif mu == 3:
        out = mu_3(X, K)


    # Add the noise to the output
    df["noise"] = noise
    df["Y"] = out + noise

    # Generate 5 fake variables
    df["fake"] = norm.rvs(loc=0, scale=1, size=(n * k, 1))
    df["fake2"] = norm.rvs(loc=0, scale=1, size=(n * k, 1))
    df["fake3"] = norm.rvs(loc=0, scale=1, size=(n * k, 1))
    df["fake4"] = norm.rvs(loc=0, scale=1, size=(n * k, 1))
    df["fake5"] = norm.rvs(loc=0, scale=1, size=(n * k, 1))

    # Standardize Y
    df["Y"] = (df["Y"] - df["Y"].mean()) / df["Y"].std()

    # Output final dataframe
    return df




def main():

    # Initialize list to hold outputs
    results_list = []

    # Iterate over n values
    for n in [20, 50, 100, 500]:

        # Iterate over seeds
        for seed in range(num_seeds):

            # Generate data
            data = generate_data(
                seed=seed,
                p=p,
                n=n,
                k=k,
                noise_sigma=noise_sigma,
                mu = mu,
            )

            ## Uncomment line if you would like to save the generated data
            # data.to_csv("./syn_data/data_n-" + str(n) + "-Seed-" + str(seed) + "-mu-" + str(mu) + ".csv", index=False)

            # Train-test split
            train_ratio = 0.8
            last_group = int(k * train_ratio)
            X_train = data[data["group"] < last_group]
            X_test = data[data["group"] >= last_group]

            input_vars = ["X1", "X2", "X3", "X4", "X5", "fake", "fake2", "fake3", "fake4", "fake5"]

            ########################################################################################

            ## Regular decision tree
            start_time = time()
            tree = DTR()
            tree.fit(X_train[input_vars], X_train["Y"])
            pred = tree.predict(X_test[input_vars])
            tree_mse = mse(X_test["Y"], pred)
            results_list.append(
                {
                    "method": "Decision Tree",
                    "MSE": tree_mse,
                    "n": str(n),
                    "seed": seed,
                    "time": time() - start_time,
                }
            )

            ########################################################################################

            ## Regular random forest
            start_time = time()
            tree = RFR(n_estimators=last_group)
            tree.fit(X_train[input_vars], X_train["Y"])
            pred = tree.predict(X_test[input_vars])
            forest_mse = mse(X_test["Y"], pred)
            results_list.append(
                {
                    "method": "Random Forest",
                    "MSE": forest_mse,
                    "n": str(n),
                    "seed": seed,
                    "time": time() - start_time,
                }
            )

            ########################################################################################

            ## Regular Linear Mixed Model
            start_time = time()
            df = X_train.drop(columns=["noise", "group"])
            form = "X1 + X2 + X3 + X4 + X5 + fake" # X6 + X7 + X8 + X9 + X10"
            form = form + " + fake2 + fake3 + fake4 + fake5"
            md = smf.mixedlm("Y ~ " + form, df, groups=X_train["group"], re_formula=form)
            mdf = md.fit()
            pred = mdf.predict(X_test)
            lmm_mse = mse(X_test["Y"], pred)
            results_list.append(
                {
                    "method": "Linear Mixed Model",
                    "MSE": lmm_mse,
                    "n": str(n),
                    "seed": seed,
                    "time": time() - start_time,
                }
            )

            ########################################################################################

            # Weighted Sum-of-Trees

            # Build group classifier
            start_time = time()
            group_clf = LR() # GNB()
            group_clf.fit(
                X_train[input_vars], X_train["group"]
            )
            group_pred = group_clf.predict_proba(
                X_test[input_vars]
            )

            # Normalize group predictions
            row_sums = group_pred.sum(axis=1)
            group_pred = group_pred / row_sums[:, np.newaxis]

            for test_group in range(last_group, k):
                rows = np.where(X_test["group"] == test_group)
                average = np.mean(group_pred[rows,], axis=1)
                group_pred[rows] = average

            # Build Sum-of-Trees
            list_of_trees = []
            for i in range(last_group):
                tree = DTR()
                tree.fit(
                    X_train[X_train["group"] == i][
                        input_vars
                    ],
                    X_train[X_train["group"] == i]["Y"],
                )
                list_of_trees.append(tree)

            preds = np.array(
                [
                    tree.predict(X_test[input_vars])
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
                    "n": str(n),
                    "seed": seed,
                    "time": time() - start_time,
                }
            )

            ########################################################################################

            ## Weighted Sum-of-Forests
            start_time = time()
            list_of_trees = []
            for i in range(last_group):
                tree = RFR(n_estimators=last_group)
                # tree = RFR()
                tree.fit(
                    X_train[X_train["group"] == i][
                        input_vars
                    ],
                    X_train[X_train["group"] == i]["Y"],
                )
                list_of_trees.append(tree)

            preds = np.array(
                [
                    tree.predict(X_test[input_vars])
                    for tree in list_of_trees
                ]
            )
            preds = preds.T
            num = preds.shape[0]
            pred = [np.dot(preds[i, :], group_pred[i, :]) for i in range(num)]
            my_mse = mse(X_test["Y"], pred)
            results_list.append(
                {
                    "method": "Weighted Sum-of-Forests",
                    "MSE": my_mse,
                    "n": str(n),
                    "seed": seed,
                    "time": time() - start_time,
                }
            )

            ########################################################################################

        print("Finished TEST n=", n, "\n")

    # Save results in csv and plot
    df = pd.DataFrame(results_list)
    curr_datetime = strftime(f"%Y-%m-%d_%H-%M", localtime())
    filename = "./out/" + curr_datetime + test_name 
    df.to_csv(filename + ".csv", index=False)

    # Plot results
    sns.boxplot(
        data=df,
        x="n",
        y="MSE",
        hue="method",
        palette="Paired",
        hue_order=[
            "Linear Mixed Model",
            "Decision Tree",
            "Random Forest",
            "Weighted Sum-of-Trees",
            "Weighted Sum-of-Forests",
        ],
    )
    plt.rcParams['text.usetex'] = True
    plt.xlabel(r'$n$')
    plt.savefig(filename + ".png", dpi=600)

    # Save parameters
    with open(filename + ".txt", "a") as f:
        print("datetime:", curr_datetime, file=f)
        print("p: ", p, file=f)
        print("k: ", k, file=f)
        print("mu: ", mu, file=f)
        print("noise_sigma: ", noise_sigma, file=f)
        print("num_seeds: ", num_seeds, file=f)
        print("test_name: ", test_name, file=f)
        print("Group CLF:", group_clf, file=f)


###############################################################################
############################## MAIN FUNCTION ##################################
###############################################################################

if __name__ == "__main__":

    test_name = "mu_1"

    # Simulated Data parameters
    p = 5
    mu = 1
    k = 20
    noise_sigma = 1
    num_seeds = 5

    print("...starting...\n")

    main()

    print("Simulation complete.")
