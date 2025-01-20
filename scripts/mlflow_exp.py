import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score



# Set the experiment name (it will be created if it doesn't exist)
experiment_name = "Iris_Model_Experiment"
mlflow.set_experiment(experiment_name)

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the RandomForestClassifier
clf = RandomForestClassifier()

# Start MLflow experiment
with mlflow.start_run():
    # Train the model
    clf.fit(X_train, y_train)

    # Make predictions
    y_pred = clf.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Log the accuracy metric
    mlflow.log_metric("accuracy", accuracy)

    # Log the model with input_example for signature
    mlflow.sklearn.log_model(clf, "model", input_example=X_test[:1])  # Using first example for input

    print(f"Accuracy: {accuracy}")
