from necessaryImports import *
from gettingDataReady import *

# Create a hyperparameter grid for RandomForestClassifier
rf_grid = {"n_estimators": np.arange(10,1000,50),
           "max_depth": [None, 3, 5, 10],
           "min_samples_split": np.arange(2, 20, 2),
           "min_samples_leaf": np.arange(1, 20, 2)}

# Setup random hyperparameter search for RandomForestClassifier
rs_rf = RandomizedSearchCV(RandomForestClassifier(),
                           param_distributions=rf_grid,
                           cv=5,
                           n_iter=20,
                           verbose=True,
                           n_jobs=-1)

# Fit random Hyperparameter search model for RandomForestClassifier()       
rs_rf.fit(X_train, y_train)

# Evaluate the randomized search RandomForestClasifier model
print(rs_rf.score(X_test, y_test))