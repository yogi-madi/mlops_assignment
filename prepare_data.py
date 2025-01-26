# FILE: prepare_data.py
import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset
data = pd.read_csv('data/breast_cancer.csv')

# Split the dataset into features and target
X = data.drop('target', axis=1)
y = data['target']

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=42)

# Save the training and test sets
X_train.to_csv('data/X_train.csv', index=False)
X_test.to_csv('data/X_test.csv', index=False)
y_train.to_csv('data/y_train.csv', index=False)
y_test.to_csv('data/y_test.csv', index=False)
