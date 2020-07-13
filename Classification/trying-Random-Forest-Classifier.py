from necessaryImports import *

np.random.seed(42)

db = pd.read_csv("heart-disease.csv")

# Prepraring data to X and y
X = db.drop("target", axis=1)
y = db["target"]

# Spliting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


clf = RandomForestClassifier()
clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))