import numpy as np

# Define X and y
X = np.array([
    [1, 1],
    [1, 4]
])

y = np.array([3, 9])

# Compute inverse of X
X_inv = np.linalg.inv(X)

# Compute theta
theta = X_inv @ y

# Display results
print("X inverse:")
print(X_inv)
print("\nTheta:")
print(theta)

# Optional: confirm prediction
def predict(x):
    return theta[0] + theta[1] * x

print("\nPrediction for x=1:", predict(1))
print("Prediction for x=4:", predict(4))
