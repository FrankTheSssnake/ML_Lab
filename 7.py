print("Aaradhya Bhardwaj 24/SE/005")
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pandas as pd
import numpy as np

data = load_diabetes()
df = pd.DataFrame(data.data, columns=data.feature_names)
print(df.head())
print(df.shape)

X = df
y = (data.target > data.target.mean()).astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
print(np.unique(y_train))
model.fit(X_train, y_train)

def predict_sample(sample):
    sample_df = pd.DataFrame([sample], columns=df.columns)
    sample_scaled = sc.transform(sample_df)
    return model.predict(sample_scaled)

y_pred = model.predict(X_test)
print(y_pred[:10])

print("Unique classes:", np.unique(y_pred))
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

print(classification_report(y_test, y_pred))

sample = X_test[0]
print("Predicted Class:", model.predict([sample])[0])
print("Actual Class:", y_test[0])
