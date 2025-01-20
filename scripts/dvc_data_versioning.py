import pandas as pd
from sklearn.datasets import load_iris

# Load the dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target

# Save the dataset to a CSV file
df.to_csv('data/iris.csv', index=False)
