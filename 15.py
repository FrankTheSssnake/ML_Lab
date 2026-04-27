print("Aaradhya Bhardwaj 24/SE/005")
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

data = load_breast_cancer()
X = data.data
y = data.target
print("Dataset shape:", X.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(int)

input_layer = X_train.shape[1]
hidden_layer1 = 10
hidden_layer2 = 5
output_layer = 1
np.random.seed(42)
W1 = np.random.randn(input_layer, hidden_layer1)
b1 = np.zeros((1, hidden_layer1))
W2 = np.random.randn(hidden_layer1, hidden_layer2)
b2 = np.zeros((1, hidden_layer2))
W3 = np.random.randn(hidden_layer2, output_layer)
b3 = np.zeros((1, output_layer))

epochs = 1000
learning_rate = 0.01
y_train = y_train.reshape(-1, 1)
for epoch in range(epochs):
    z1 = np.dot(X_train, W1) + b1
    a1 = relu(z1)
    z2 = np.dot(a1, W2) + b2
    a2 = relu(z2)
    z3 = np.dot(a2, W3) + b3
    output = sigmoid(z3)
    error = y_train - output
    d_output = error * sigmoid_derivative(output)
    error2 = d_output.dot(W3.T)
    d_hidden2 = error2 * relu_derivative(a2)
    error1 = d_hidden2.dot(W2.T)
    d_hidden1 = error1 * relu_derivative(a1)
    W3 += a2.T.dot(d_output) * learning_rate
    b3 += np.sum(d_output, axis=0, keepdims=True) * learning_rate
    W2 += a1.T.dot(d_hidden2) * learning_rate
    b2 += np.sum(d_hidden2, axis=0, keepdims=True) * learning_rate
    W1 += X_train.T.dot(d_hidden1) * learning_rate
    b1 += np.sum(d_hidden1, axis=0, keepdims=True) * learning_rate
print("Training Completed")

def predict(X):
    a1 = relu(np.dot(X, W1) + b1)
    a2 = relu(np.dot(a1, W2) + b2)
    output = sigmoid(np.dot(a2, W3) + b3)
    return (output > 0.5).astype(int)

y_pred = predict(X_test)
accuracy = np.mean(y_pred.flatten() == y_test)
print("Accuracy:", accuracy)

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))

sample = X_test[0].reshape(1, -1)
prediction = predict(sample)
print("Actual:", y_test[0])
print("Predicted:", prediction[0][0])
