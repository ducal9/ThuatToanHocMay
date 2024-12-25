from sklearn.metrics import silhouette_score
labels = data['Cluster']
score = silhouette_score(data_scaled, labels)
print(f"Silhouette Score: {score}")
