print("Aaradhya Bhardwaj 24/SE/005")
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

data = load_diabetes()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target
print(df.head())
print(df.shape)

X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
print(X_train.shape, X_test.shape)

X_train = (X_train - X_train.mean()) / X_train.std()
X_test = (X_test - X_test.mean()) / X_test.std()

m = X_train.shape[0]
n = X_train.shape[1]
weights = np.zeros(n)
bias = 0
learning_rate = 0.01
epochs = 1000

for i in range(epochs):
    y_pred = np.dot(X_train, weights) + bias
    error = y_pred - y_train
    dw = (1 / m) * np.dot(X_train.T, error)
    db = (1 / m) * np.sum(error)
    weights = weights - learning_rate * dw
    bias = bias - learning_rate * db
print("Training Complete")

y_pred_test = np.dot(X_test, weights) + bias
mse = np.mean((y_pred_test - y_test) ** 2)
mae = np.mean(np.abs(y_pred_test - y_test))
print("MSE:", mse)
print("MAE:", mae)

sample = X_test.iloc[0]
prediction = np.dot(sample, weights) + bias
print("Actual:", y_test.iloc[0])
print("Predicted:", prediction)
