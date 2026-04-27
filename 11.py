print("Aaradhya Bhardwaj 24/SE/005")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(10)
data1 = np.random.randn(50, 2) + [2, 3]
data2 = np.random.randn(50, 2) + [7, 5]
data3 = np.random.randn(50, 2) + [4, 9]
dataset = np.vstack((data1, data2, data3))
df = pd.DataFrame(dataset, columns=['X', 'Y'])
print(df.head())

k = 3
centroids = df.sample(k).values

def distance(p1, p2):
    return np.sqrt(np.sum((p1 - p2) ** 2))

max_iterations = 100
for itr in range(max_iterations):
    cluster_points = [[] for _ in range(k)]
    for point in df.values:
        dist = [distance(point, c) for c in centroids]
        index = np.argmin(dist)
        cluster_points[index].append(point)
    new_centroids = []
    for cluster in cluster_points:
        new_centroids.append(np.mean(cluster, axis=0))
    new_centroids = np.array(new_centroids)
    if np.all(centroids == new_centroids):
        break
    centroids = new_centroids

colors = ['red', 'blue', 'green']
for i in range(k):
    cluster = np.array(cluster_points[i])
    plt.scatter(cluster[:, 0], cluster[:, 1], color=colors[i])
plt.scatter(centroids[:, 0], centroids[:, 1], color='black', marker='X', s=200)
plt.title("K-Means Clustering")
plt.xlabel("X values")
plt.ylabel("Y values")
plt.show()

wcss = 0
for i in range(k):
    cluster = np.array(cluster_points[i])
    for point in cluster:
        wcss += distance(point, centroids[i]) ** 2
print("WCSS Value:", wcss)
