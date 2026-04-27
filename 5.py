print("Aaradhya Bhardwaj 24/SE/005")
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import pandas as pd
import numpy as np

data = load_diabetes()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target
df['target'] = np.where(df['target'] > df['target'].median(), 1, 0)
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
epochs = 5000

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

for i in range(epochs):
    z = np.dot(X_train, weights) + bias
    y_pred = sigmoid(z)
    error = y_pred - y_train
    dw = (1 / m) * np.dot(X_train.T, error)
    db = (1 / m) * np.sum(error)
    weights = weights - learning_rate * dw
    bias = bias - learning_rate * db
print("Training Complete")

def predict(X):
    z = np.dot(X, weights) + bias
    y_pred = sigmoid(z)
    return [1 if i > 0.5 else 0 for i in y_pred]

predictions = predict(X_test)
accuracy = np.mean(predictions == y_test)
print("Accuracy:", accuracy)

cm = confusion_matrix(y_test, predictions)
print("Confusion Matrix:")
print(cm)

precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)
f1 = f1_score(y_test, predictions)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

sample = X_test.iloc[0]
prediction = predict([sample])
print("Actual:", y_test.iloc[0])
print("Predicted:", prediction[0])
