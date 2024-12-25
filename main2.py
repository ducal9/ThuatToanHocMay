import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Đọc dữ liệu
file_path = 'gmm_sample_data.csv'  # Thay bằng đường dẫn file của bạn
data = pd.read_csv(file_path)

# 2. Chuẩn hóa dữ liệu
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

# 3. Xác định số cụm tối ưu bằng Silhouette Score
silhouette_scores = []
cluster_range = range(2, 10)  # Thử nghiệm từ 2 đến 9 cụm

for n_clusters in cluster_range:
    gmm = GaussianMixture(n_components=n_clusters, random_state=42)
    cluster_labels = gmm.fit_predict(scaled_data)
    score = silhouette_score(scaled_data, cluster_labels)
    silhouette_scores.append(score)

# 4. Vẽ biểu đồ Silhouette để chọn số cụm
plt.figure(figsize=(10, 6))
plt.plot(cluster_range, silhouette_scores, marker='o')
plt.title("Silhouette Scores for GMM")
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Score")
plt.show()

# 5. Xây dựng mô hình GMM với số cụm tối ưu
optimal_clusters = cluster_range[silhouette_scores.index(max(silhouette_scores))]
print(f"Số cụm tối ưu: {optimal_clusters}")

gmm = GaussianMixture(n_components=optimal_clusters, random_state=42)
data['Cluster'] = gmm.fit_predict(scaled_data)

# 6. Trực quan hóa dữ liệu (nếu có 2 hoặc 3 đặc trưng)
if data.shape[1] - 1 <= 3:  # Trừ cột 'Cluster'
    sns.pairplot(data, hue='Cluster', palette='tab10')
    plt.show()

# 7. Lưu kết quả vào file CSV
data.to_csv('gmm_clustered_data.csv', index=False)
print("Kết quả phân cụm đã được lưu vào 'gmm_clustered_data.csv'.")
