from necessaryImports import *
from gettingDataReady import *

# Create a hyperparameter grid for LogiticRegresion
log_reg_grid={"C": np.logspace(-4, 4, 20),
              "solver": ["liblinear"]}

# Setup random hyperparameter search for LogisticRegression
rs_log_reg = RandomizedSearchCV(LogisticRegression(),
                                param_distributions=log_reg_grid,
                                cv = 5,
                                n_iter=20,
                                verbose=True,
                                n_jobs=-1)

# Fit random hyperparameters search model for Logistic Regression
rs_log_reg.fit(X_train, y_train)

print(rs_log_reg.score(X_test, y_test))