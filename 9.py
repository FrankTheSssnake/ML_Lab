print("Aaradhya Bhardwaj 24/SE/005")
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from math import log2

data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target
print(df.head())
print("\nShape of dataset:", df.shape)

for column in df.columns[:-1]:
    df[column] = pd.qcut(df[column], q=3, labels=["Low", "Medium", "High"])
print(df.head())

def entropy(data):
    values = data.value_counts(normalize=True)
    ent = 0
    for v in values:
        if v > 0:
            ent -= v * log2(v)
    return ent

def information_gain(data, attribute, target):
    total_entropy = entropy(data[target])
    values = data[attribute].unique()
    weighted_entropy = 0
    for val in values:
        subset = data[data[attribute] == val]
        weight = len(subset) / len(data)
        weighted_entropy += weight * entropy(subset[target])
    gain = total_entropy - weighted_entropy
    return gain

def split_info(data, attribute):
    values = data[attribute].value_counts(normalize=True)
    split = 0
    for v in values:
        if v > 0:
            split -= v * log2(v)
    return split

def gain_ratio(data, attribute, target):
    ig = information_gain(data, attribute, target)
    sp = split_info(data, attribute)
    if sp == 0:
        return 0
    return ig / sp

def c45(data, features, target):
    if len(data[target].unique()) == 1:
        return data[target].iloc[0]
    if len(features) == 0:
        return data[target].mode()[0]
    ratios = []
    for feature in features:
        ratios.append(gain_ratio(data, feature, target))
    best_feature = features[np.argmax(ratios)]
    tree = {best_feature: {}}
    for value in data[best_feature].unique():
        subset = data[data[best_feature] == value]
        remaining_features = [f for f in features if f != best_feature]
        subtree = c45(subset, remaining_features, target)
        tree[best_feature][value] = subtree
    return tree

train, test = train_test_split(df, test_size=0.3, random_state=10)
print("Training size:", train.shape)
print("Testing size:", test.shape)

features = list(train.columns[:-1])
target = 'target'
tree = c45(train, features, target)

def predict(sample, tree):
    if not isinstance(tree, dict):
        return tree
    root = list(tree.keys())[0]
    value = sample[root]
    subtree = tree[root].get(value)
    return predict(sample, subtree)

correct = 0
for i in range(len(test)):
    pred = predict(test.iloc[i], tree)
    if pred == test.iloc[i]['target']:
        correct += 1
accuracy = correct / len(test)
print("Accuracy:", accuracy)

sample = test.iloc[0]
prediction = predict(sample, tree)
print("Sample:")
print(sample)
print("\nPredicted Class:", prediction)
