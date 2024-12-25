from sklearn.cluster import OPTICS
import pandas as pd
import matplotlib.pyplot as plt

# Bước 1: Đọc dữ liệu từ file CSV
file_path = "optics_data.csv"
data = pd.read_csv(file_path)

# Bước 2: Áp dụng thuật toán OPTICS
# Cấu hình các tham số
optics_model = OPTICS(min_samples=10, xi=0.1, min_cluster_size=0.05)
optics_model.fit(data)

# Gắn nhãn cụm vào dữ liệu
data['Cluster'] = optics_model.labels_

# Bước 3: Trực quan hóa kết quả
plt.figure(figsize=(8, 6))
plt.scatter(data['Feature1'], data['Feature2'], c=data['Cluster'], cmap='rainbow', s=10)
plt.title("OPTICS Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.colorbar(label="Cluster")
plt.show()

# Lưu kết quả vào file CSV
output_file = "optics_clusters.csv"
data.to_csv(output_file, index=False)
print(f"Kết quả phân cụm đã được lưu vào '{output_file}'")
