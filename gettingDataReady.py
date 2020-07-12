from necessaryImports import *

# Creating a Random seed
np.random.seed(42)

# Importing the data into a DataFrame
df = pd.read_csv("heart-disease.csv")

# Getting the data ready into X and y
X = df.drop("target", axis=1)
y = df["target"]

# Splitting the data into train and test datas
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
