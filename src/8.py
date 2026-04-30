print("Aaradhya Bhardwaj 24/SE/005")
from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
import math

data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target
print(df.head())

for col in data.feature_names:
    df[col] = pd.cut(df[col], bins=2, labels=[0, 1])
print(df.head())

def entropy(col):
    counts = col.value_counts()
    total = len(col)
    ent = 0
    for c in counts:
        p = c / total
        ent += -p * math.log2(p)
    return ent

def information_gain(df, feature, target):
    total_entropy = entropy(df[target])
    values = df[feature].unique()
    weighted_entropy = 0
    for v in values:
        subset = df[df[feature] == v]
        weighted_entropy += (len(subset) / len(df)) * entropy(subset[target])
    return total_entropy - weighted_entropy

def id3(df, features, target):
    if len(df[target].unique()) == 1:
        return df[target].iloc[0]
    if len(features) == 0:
        return df[target].mode()[0]
    gains = [information_gain(df, f, target) for f in features]
    best_feature = features[np.argmax(gains)]
    tree = {best_feature: {}}
    for val in df[best_feature].unique():
        subset = df[df[best_feature] == val]
        subtree = id3(subset, [f for f in features if f != best_feature], target)
        tree[best_feature][val] = subtree
    return tree

features = list(data.feature_names)
target = 'target'
tree = id3(df, features, target)
print("Decision Tree:\n")
print(tree)

def predict(tree, sample):
    if not isinstance(tree, dict):
        return tree
    feature = list(tree.keys())[0]
    value = sample[feature]
    if value in tree[feature]:
        return predict(tree[feature][value], sample)
    else:
        return 0

correct = 0
for i in range(len(df)):
    sample = df.iloc[i].drop('target')
    pred = predict(tree, sample)
    if pred == df['target'].iloc[i]:
        correct += 1
accuracy = correct / len(df)
print("Accuracy:", accuracy)

sample = df.iloc[0].drop('target')
prediction = predict(tree, sample)
print("Predicted Class:", prediction)
print("Actual Class:", df['target'].iloc[0])
