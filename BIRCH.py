from sklearn.cluster import Birch
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt

# Bước 1: Đọc dữ liệu từ file CSV
file_path = "birch_data.csv"
data = pd.read_csv(file_path)

# Chuẩn hóa dữ liệu (StandardScaler đảm bảo dữ liệu đồng nhất)
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

# Bước 2: Áp dụng thuật toán BIRCH
n_clusters = 4  # Số cụm mong muốn
birch_model = Birch(n_clusters=n_clusters)
labels = birch_model.fit_predict(scaled_data)

# Thêm nhãn cụm vào dữ liệu gốc
data['Cluster'] = labels

# Bước 3: Trực quan hóa dữ liệu
# Chỉ hiển thị Feature1 và Feature2 để vẽ Scatter Plot
plt.figure(figsize=(8, 6))
plt.scatter(data['Feature1'], data['Feature2'], c=data['Cluster'], cmap='viridis', s=30)
plt.title("BIRCH Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.colorbar(label="Cluster")
plt.show()

# Lưu kết quả phân cụm vào file CSV
output_file = "birch_clusters.csv"
data.to_csv(output_file, index=False)
print(f"Kết quả phân cụm đã được lưu vào '{output_file}'")
