import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import joblib
import os
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

# Set the MLflow tracking URI (optional, defaults to ./mlruns)
mlflow.set_tracking_uri("mlruns")

# Set the experiment name
mlflow.set_experiment("mlops")

# Load the training data
X_train = pd.read_csv('data/X_train.csv')
y_train = pd.read_csv('data/y_train.csv')

# Define the parameter grid for GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 15]
}

# Initialize the RandomForest model
model = RandomForestClassifier(random_state=42)

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy')

# Start an MLflow run
with mlflow.start_run():
    # Fit GridSearchCV
    grid_search.fit(X_train, y_train.values.ravel())

    # Get the best model
    best_model = grid_search.best_estimator_

    # Log the best parameters
    mlflow.log_params(grid_search.best_params_)

    # Log the best model metrics
    best_accuracy = grid_search.best_score_
    mlflow.log_metric("best_accuracy", best_accuracy)

    # Infer model signature
    signature = infer_signature(X_train, best_model.predict(X_train))

    # Log the best model with signature and input example
    mlflow.sklearn.log_model(best_model, "model", signature=signature, input_example=X_train.head())

    # Create the model directory if it doesn't exist
    os.makedirs('model', exist_ok=True)

    # Save the best model
    joblib.dump(best_model, 'models/best_model.pkl')