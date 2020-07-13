import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV


df = pd.read_csv("heart-disease.csv")
# Split data into X and y
X = df.drop("target", axis=1)

y = df["target"]

# Split data into Train and test sets
np.random.seed(100)

# Split into train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create a hyperparameter grid for RandomForestClassifier
rf_grid = {"n_estimators": np.arange(10,1000,50),
           "max_depth": [None, 3, 5, 10],
           "min_samples_split": np.arange(2, 20, 2),
           "min_samples_leaf": np.arange(1, 20, 2)}
# Tune Logitic Regression

np.random.seed(100)

# Setup random hyperparameter search for RandomForestClassifer
rs_log_reg = RandomizedSearchCV(RandomForestClassifier(),
                                param_distributions=rf_grid,
                                cv = 5,
                                n_iter=20,
                                verbose=True,
                                n_jobs=-1)

# Fit random hyperparameters search model for Logistic Regression
rs_log_reg.fit(X_train, y_train)

print(rs_log_reg.best_params_)
print(rs_log_reg.score(X_test, y_test))