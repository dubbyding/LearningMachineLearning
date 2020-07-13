from necessaryImports import *
from gettingDataReady import *

# Put models in a dictionary
models = {"Logistic Regression": LogisticRegression(),
          "KNN": KNeighborsClassifier(),
          "Random Forest": RandomForestClassifier()}

# Make a dictionary to keep model scores
model_scores = {}
# Loop through models
for name, model in models.items():
    # Fit the model to the data
    model.fit(X_train, y_train)
    # Evaluate the model and append it's score to model_scores
    model_scores[name] = model.score(X_test, y_test)
print(model_scores)