from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Bước 1: Đọc dữ liệu từ file CSV
file_path = "data_cuoiky.csv"
data = pd.read_csv(file_path)

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Bước 2: Áp dụng DBSCAN
eps = 3.3  # Bán kính lân cận
min_samples = 1  # Số điểm tối thiểu để tạo cụm
dbscan = DBSCAN(eps=eps, min_samples=min_samples)
data['Cluster'] = dbscan.fit_predict(data_scaled)

# Bước 3: Giảm chiều để trực quan hóa
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(data_scaled)

# Bước 4: Trực quan hóa cụm
plt.figure(figsize=(10, 8))
sns.scatterplot(x=reduced_data[:, 0], y=reduced_data[:, 1], hue=data['Cluster'], palette='Set2', s=50, alpha=0.8)
plt.title("DBSCAN Clustering")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend(title="Cluster")
plt.show()

# Bước 5: Lưu kết quả vào file CSV
output_file = "dbscan_advanced_clusters.csv"
data.to_csv(output_file, index=False)
print(f"Kết quả phân cụm đã được lưu vào '{output_file}'")
