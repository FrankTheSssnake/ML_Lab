print("Aaradhya Bhardwaj 24/SE/005")
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import pandas as pd
import numpy as np

data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target
print(df.head())
print(df.shape)

X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
print(X_train.shape, X_test.shape)

def euclidean(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

def predict(X_train, y_train, test, k=3):
    distances = [euclidean(test, x) for x in X_train]
    idx = np.argsort(distances)[:k]
    labels = y_train.iloc[idx]
    return labels.mode()[0]

predictions = []
for i in range(len(X_test)):
    pred = predict(X_train.values, y_train, X_test.iloc[i].values)
    predictions.append(pred)
print(predictions[:10])

correct = 0
wrong = 0
for i in range(len(predictions)):
    if predictions[i] == y_test.iloc[i]:
        correct += 1
    else:
        wrong += 1
accuracy = correct / len(predictions)
print("Correct Predictions:", correct)
print("Wrong Predictions:", wrong)
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
prediction = predict(X_train.values, y_train, sample.values)
print("Actual:", y_test.iloc[0])
print("Predicted:", prediction)
