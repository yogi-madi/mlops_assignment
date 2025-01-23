import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load the dataset
df = pd.read_csv("data/breast_cancer.csv")
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize model
model = RandomForestClassifier(random_state=42)

# Define parameter grid
param_grid = {
    "n_estimators": [50, 100, 150],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5, 10]
}

# Perform grid search
grid_search = GridSearchCV(model, param_grid, cv=5, scoring="accuracy")
grid_search.fit(X_train, y_train)

# Best parameters and accuracy
best_model = grid_search.best_estimator_
best_params = grid_search.best_params_
best_score = grid_search.best_score_

# Save the model
joblib.dump(best_model, "models/best_model.pkl")

# Save results
with open("results/hyperparameter_tuning_results.md", "w") as f:
    f.write(f"Best Parameters: {best_params}\n")
    f.write(f"Best Cross-Validation Score: {best_score}\n")

# Test the model
y_pred = best_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
with open("results/hyperparameter_tuning_results.md", "a") as f:
    f.write(f"Test Accuracy: {test_accuracy}\n")
