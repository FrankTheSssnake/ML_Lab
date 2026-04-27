print("Aaradhya Bhardwaj 24/SE/005")
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold, chi2, RFE, SequentialFeatureSelector
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np

data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target
X = df.drop('target', axis=1)
y = df['target']
X = X.fillna(X.mean())
X_scaled = (X - X.mean()) / X.std()
z = np.abs((X_scaled - X_scaled.mean()) / X_scaled.std())
X_clean = X_scaled[(z < 3).all(axis=1)]
y_clean = y[X_clean.index]
df_clean = X_clean.copy()
df_clean['target'] = y_clean
print(df_clean.shape)

corr = df_clean.corr()
target_corr = corr['target'].abs()
selected_features = target_corr[target_corr > 0.2].index
print("Selected Features:")
print(selected_features)

X = df_clean.drop('target', axis=1)
selector = VarianceThreshold(threshold=0.01)
selector.fit_transform(X)
selected = X.columns[selector.get_support()]
print("Selected Features:")
print(selected)

X = df_clean.drop('target', axis=1)
y = df_clean['target']
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
chi_scores = chi2(X_scaled, y)
scores = pd.Series(chi_scores[0], index=X.columns)
selected = scores.sort_values(ascending=False).head(10)
print(selected)

model = LogisticRegression(max_iter=5000)
rfe = RFE(model, n_features_to_select=10)
rfe.fit(X, y)
selected = X.columns[rfe.support_]
print(selected)

model = KNeighborsClassifier()
sfs = SequentialFeatureSelector(model, n_features_to_select=10, direction='forward')
sfs.fit(X, y)
selected = X.columns[sfs.get_support()]
print(selected)

model = KNeighborsClassifier()
sfs = SequentialFeatureSelector(model, n_features_to_select=10, direction='backward')
sfs.fit(X, y)
selected = X.columns[sfs.get_support()]
print(selected)

X = df_clean.drop('target', axis=1)
y = df_clean['target']
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
