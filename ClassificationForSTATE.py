#libraries needed
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
import scipy.stats as st
from sklearn.model_selection import KFold
from sklearn.dummy import DummyClassifier
from sklearn.metrics import zero_one_loss
from dtuimldmtools import *
from dtuimldmtools.statistics.statistics import correlated_ttest
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import LeaveOneGroupOut
from statsmodels.stats.contingency_tables import mcnemar
from scipy.stats import binom
from scipy.stats import chi2, binom, beta

my_file = '/Users/emilehourmanditlefsen/Downloads/HR_Xnew.csv'
data = pd.read_csv(my_file)

#Feature and target
attributes = ['HR_Mean', 'HR_Median', 'HR_std', 'HR_Min', 'HR_Max', 'HR_AUC']
X = data[attributes].values
y = data['Frustrated'].values
groups = data['Individual'].values


unique_individuals = np.unique(groups)
print(f"Number of unique individuals: {len(unique_individuals)}")

LGO = LeaveOneGroupOut()

#KNN and random forest
k_values = range(1, 13) #range for Nearest neighbors
n_estimators_values = range(10, 201, 10)  #range for estimators

check_var = False #For skipping the CV in the beginnning

#CV Loop
if check_var == True:
    for fold, (train_index, test_index) in enumerate(LGO.split(X, y, groups), 1):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        #Standardization
        mu = np.mean(X_train, axis=0)
        sigma = np.std(X_train, axis=0)
        X_train_std = (X_train - mu) / sigma
        X_test_std = (X_test - mu) / sigma

        #Baseline
        most_frequent_class = np.argmax(np.bincount(y_train))
        baseline_predictions = np.full(shape=y_test.shape, fill_value=most_frequent_class)
        baseline_error = np.mean(baseline_predictions != y_test)

        #RF
        fold_test_error_rates_rf = np.zeros(len(n_estimators_values))
        for idx, n_estimators in enumerate(n_estimators_values):
            rf_model = RandomForestClassifier(n_estimators=n_estimators, random_state=1)
            rf_model.fit(X_train_std, y_train)
            fold_test_error_rates_rf[idx] = np.mean(rf_model.predict(X_test_std) != y_test)
        min_error_rf = np.min(fold_test_error_rates_rf)
        opt_n_estimators_idx = np.argmin(fold_test_error_rates_rf)
        opt_n_estimators = n_estimators_values[opt_n_estimators_idx]

        #KNN
        fold_test_error_rates_knn = np.zeros(len(k_values))
        for idx, k in enumerate(k_values):
            mdl_knn = KNeighborsClassifier(n_neighbors=k)
            mdl_knn.fit(X_train_std, y_train)
            fold_test_error_rates_knn[idx] = np.mean(mdl_knn.predict(X_test_std) != y_test)
        min_error_knn = np.min(fold_test_error_rates_knn)
        opt_k_idx = np.argmin(fold_test_error_rates_knn)
        opt_k = k_values[opt_k_idx]

        #Fold result
        print(len(test_index))
        print(f"Fold {fold}:")
        print(f"Baseline Error={baseline_error * 100:.2f}%")
        print(f"Optimal Random Forest n_estimators={opt_n_estimators}, Test Error RF={min_error_rf * 100:.2f}%")
        print(f"Optimal KNN k={opt_k}, Test Error KNN={min_error_knn * 100:.2f}%")
        print()



#Models
rf_model = RandomForestClassifier(n_estimators=10, random_state=2)
baseline_model = DummyClassifier(strategy="most_frequent")
knn_model = KNeighborsClassifier(n_neighbors=2)

#LOGO
LGO = LeaveOneGroupOut()

#Pred by case
all_y_test = []
all_y_pred_rf = []
all_y_pred_baseline = []
all_y_pred_knn = []

for train_index, test_index in LGO.split(X, y, groups):
    #Split
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    #Standardization
    mu = np.mean(X_train, axis=0)
    sigma = np.std(X_train, axis=0)
    X_train_std = (X_train - mu) / sigma
    X_test_std = (X_test - mu) / sigma

    #Training
    rf_model.fit(X_train_std, y_train)
    baseline_model.fit(X_train, y_train)
    knn_model.fit(X_train_std, y_train)

    
    y_pred_rf = rf_model.predict(X_test_std)
    y_pred_baseline = baseline_model.predict(X_test)
    y_pred_knn = knn_model.predict(X_test_std)

    
    all_y_test.extend(y_test)
    all_y_pred_rf.extend(y_pred_rf)
    all_y_pred_baseline.extend(y_pred_baseline)
    all_y_pred_knn.extend(y_pred_knn)

all_y_test = np.array(all_y_test)
all_y_pred_rf = np.array(all_y_pred_rf)
all_y_pred_baseline = np.array(all_y_pred_baseline)
all_y_pred_knn = np.array(all_y_pred_knn)

def mcnemar(y_true, yhatA, yhatB, alpha=0.05): #taken from 02450 course library.
    nn = np.zeros((2, 2), dtype=int)
    nn[0, 0] = np.sum((yhatA == y_true) & (yhatB == y_true))
    nn[0, 1] = np.sum((yhatA == y_true) & (yhatB != y_true))
    nn[1, 0] = np.sum((yhatA != y_true) & (yhatB == y_true))
    nn[1, 1] = np.sum((yhatA != y_true) & (yhatB != y_true))

    n = np.sum(nn)
    n12 = nn[0, 1]
    n21 = nn[1, 0]

    thetahat = (n12 - n21) / n if n != 0 else 0
    Etheta = thetahat

    Q = n**2 * (n + 1) * (Etheta + 1) * (1 - Etheta) / (n * (n12 + n21) - (n12 - n21)**2)
    p = (Etheta + 1) * 0.5 * (Q - 1)
    q = (1 - Etheta) * 0.5 * (Q - 1)

    CI = tuple(lm * 2 - 1 for lm in beta.interval(1 - alpha, a=p, b=q))

    #Exact binom test
    if n12 + n21 <= 25:  # Use exact test for small samples
        p_value = 2 * binom.cdf(min(n12, n21), n12 + n21, 0.5)
    else:  #Chi-square test for large samples (not case here)
        chi2_stat = (abs(n12 - n21) - 1)**2 / (n12 + n21)
        p_value = chi2.sf(chi2_stat, df=1)

    print("Result of McNemar's test using alpha=", alpha)
    print("Comparison matrix n")
    print(nn)
    if n12 + n21 <= 10:
        print("Warning, n12+n21 is low: n12+n21=", (n12 + n21))

    print("Approximate 1-alpha confidence interval of theta: [thetaL, thetaU] = ", CI)
    print("p-value for two-sided test A and B have same accuracy (exact binomial test or chi-square test): p=", p_value)

    return thetahat, CI, p_value

#Baseline vs RF
theta_hat_rf, ci_rf, p_baseline_vs_rf = mcnemar(all_y_test, all_y_pred_baseline, all_y_pred_rf)

#Baseline vs KNN
theta_hat_knn, ci_knn, p_baseline_vs_knn = mcnemar(all_y_test, all_y_pred_baseline, all_y_pred_knn)

#KNN vs RF
theta_hat_knn_rf, ci_knn_rf, p_knn_vs_rf = mcnemar(all_y_test, all_y_pred_knn, all_y_pred_rf)

#mcnemar results
print("P-value and CI for baseline vs Random Forest")
print("P-value:", p_baseline_vs_rf)
print("Theta hat:", theta_hat_rf)
print("95% CI:", ci_rf)

print("P-value and CI for baseline vs KNN")
print("P-value:", p_baseline_vs_knn)
print("Theta hat:", theta_hat_knn)
print("95% CI:", ci_knn)

print("P-value and CI for KNN vs Random Forest")
print("P-value:", p_knn_vs_rf)
print("Theta hat:", theta_hat_knn_rf)
print("95% CI:", ci_knn_rf)






