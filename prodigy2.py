import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans

def plot_clusters(customer_purchase_history, cluster_labels, centroids=None, new_data=None):
    # Plotting the clusters
    colors = ['r', 'g', 'b', 'y', 'c', 'm']
    for i in range(num_clusters):
        cluster_data = customer_purchase_history[cluster_labels == i]
        plt.scatter(cluster_data[:, 0], cluster_data[:, 1], s=100, c=colors[i], label='Cluster {}'.format(i+1))

    # Plotting centroids
    if centroids is not None:
        plt.scatter(centroids[:, 0], centroids[:, 1], s=200, marker='*', c='black', label='Centroids')

    # Plotting new data points
    if new_data is not None:
        plt.scatter(new_data[:, 0], new_data[:, 1], s=100, c='orange', marker='x', label='New Data')

    plt.xlabel('Amount spent on Product 1')
    plt.ylabel('Amount spent on Product 2')
    plt.title('Customer Segmentation based on Purchase History')
    plt.legend()
    plt.show()

# Read customer purchase history from CSV file
customer_data = pd.read_csv(r'C:\Users\vigne\OneDrive\Desktop\PRODIGY\MACHINE LEARNING\Task02\history_purchase.csv')

# Convert DataFrame to numpy array
customer_purchase_history = customer_data.values

# Number of clusters (groups)
num_clusters = 3

# K-means clustering
kmeans = KMeans(n_clusters=num_clusters)
kmeans.fit(customer_purchase_history)

# Cluster labels assigned to each customer
cluster_labels = kmeans.labels_

# Visualize clusters
plot_clusters(customer_purchase_history, cluster_labels, kmeans.cluster_centers_)

# Input new data
new_data = np.array([
    [80, 60],   # New Customer 1
    [20, 90],   # New Customer 2
    [60, 30]    # New Customer 3
])

# Predict clusters for new data
new_data_cluster_labels = kmeans.predict(new_data)

# Visualize clusters including new data
plot_clusters(customer_purchase_history, cluster_labels, kmeans.cluster_centers_, new_data)

# Print predictions
for i, label in enumerate(new_data_cluster_labels):
    print("New Customer {} is in cluster {}".format(i+1, label+1))
