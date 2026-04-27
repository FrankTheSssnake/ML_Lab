print("Aaradhya Bhardwaj 24/SE/005")
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

data = load_breast_cancer()
X = data.data
y = data.target
print("Dataset Shape:", X.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

def separate_by_class(X, y):
    separated = {}
    for i in range(len(y)):
        label = y[i]
        if label not in separated:
            separated[label] = []
        separated[label].append(X[i])
    return separated

def summarize_dataset(dataset):
    summaries = [(np.mean(column), np.var(column)) for column in zip(*dataset)]
    return summaries

def summarize_by_class(separated):
    summaries = {}
    for class_value, rows in separated.items():
        summaries[class_value] = summarize_dataset(rows)
    return summaries

def calculate_prior(y):
    prior = {}
    total = len(y)
    for c in np.unique(y):
        prior[c] = np.sum(y == c) / total
    return prior

def gaussian_probability(x, mean, var):
    exponent = np.exp(-(x - mean) ** 2 / (2 * var))
    return (1 / np.sqrt(2 * np.pi * var)) * exponent

def predict_single(row, model, prior):
    probabilities = {}
    for class_value, class_summaries in model.items():
        probabilities[class_value] = prior[class_value]
        for i in range(len(class_summaries)):
            mean, var = class_summaries[i]
            probabilities[class_value] *= gaussian_probability(row[i], mean, var)
    return max(probabilities, key=probabilities.get)

def train_naive_bayes(X, y):
    separated = separate_by_class(X, y)
    model = summarize_by_class(separated)
    prior = calculate_prior(y)
    return model, prior

model, prior = train_naive_bayes(X_train, y_train)

predictions = []
for row in X_test:
    predictions.append(predict_single(row, model, prior))
predictions = np.array(predictions)

accuracy = np.mean(predictions == y_test)
print("Accuracy:", accuracy)

cm = confusion_matrix(y_test, predictions)
print("Confusion Matrix:")
print(cm)

print("Precision:", precision_score(y_test, predictions))
print("Recall:", recall_score(y_test, predictions))
print("F1 Score:", f1_score(y_test, predictions))

sample = X_test[0]
prediction = predict_single(sample, model, prior)
print("Actual:", y_test[0])
print("Predicted:", prediction)
