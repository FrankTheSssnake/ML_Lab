print("Aaradhya Bhardwaj 24/SE/005")
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score

data = load_breast_cancer()
X = data.data
y = data.target
print("Dataset shape:", X.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

base_model = DecisionTreeClassifier()
bagging_model = BaggingClassifier(estimator=base_model, n_estimators=50, random_state=42)
bagging_model.fit(X_train, y_train)
bagging_pred = bagging_model.predict(X_test)

ada_model = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1),
                                n_estimators=50, learning_rate=1, random_state=42)
ada_model.fit(X_train, y_train)
ada_pred = ada_model.predict(X_test)

bagging_acc = accuracy_score(y_test, bagging_pred)
ada_acc = accuracy_score(y_test, ada_pred)
print("Bagging Accuracy:", bagging_acc)
print("AdaBoost Accuracy:", ada_acc)

print("Bagging Confusion Matrix")
print(confusion_matrix(y_test, bagging_pred))
print("\nAdaBoost Confusion Matrix")
print(confusion_matrix(y_test, ada_pred))

print("\nBagging Metrics")
print("Precision:", precision_score(y_test, bagging_pred))
print("Recall:", recall_score(y_test, bagging_pred))
print("F1 Score:", f1_score(y_test, bagging_pred))

print("\nAdaBoost Metrics")
print("Precision:", precision_score(y_test, ada_pred))
print("Recall:", recall_score(y_test, ada_pred))
print("F1 Score:", f1_score(y_test, ada_pred))

sample = X_test[0].reshape(1, -1)
bag_pred = bagging_model.predict(sample)
ada_pred_sample = ada_model.predict(sample)
print("Actual:", y_test[0])
print("Bagging Prediction:", bag_pred[0])
print("AdaBoost Prediction:", ada_pred_sample[0])
