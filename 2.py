print("Aaradhya Bhardwaj 24/SE/005")
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

data = load_diabetes()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target
df['target'] = np.where(df['target'] > df['target'].median(), 1, 0)
print(df.head())
print(df.shape)

df_missing = df.copy()
df_missing.iloc[0:10, 0] = np.nan
df_missing.iloc[5:15, 1] = np.nan
print(df_missing.isnull().sum())

df_filled = df_missing.fillna(df_missing.mean())
print(df_filled.isnull().sum())

df_encoded = df_filled.copy()
print(df_encoded.head())

df_standard = (df_encoded - df_encoded.mean()) / df_encoded.std()
print(df_standard.head())

df_norm = (df_encoded - df_encoded.min()) / (df_encoded.max() - df_encoded.min())
print(df_norm.head())

z = np.abs((df_standard - df_standard.mean()) / df_standard.std())
df_clean = df_standard[(z < 3).all(axis=1)]
print(df_clean.shape)

X = df_norm.drop('target', axis=1)
y = df_norm['target']
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
accuracy = np.mean(predictions == y_test.values)
print("Accuracy:", accuracy)

for i in range(5):
    print("Actual:", y_test.iloc[i], "Predicted:", predictions[i])
