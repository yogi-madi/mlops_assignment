from sklearn.linear_model import LinearRegression
import numpy as np

# Sample data
X = np.array([[1], [2], [3]])
y = np.array([1, 2, 3])

model = LinearRegression()
model.fit(X, y)


# print
print("Model trained successfully!")
