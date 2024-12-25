import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

# Bước 1: Đọc dữ liệu từ file CSV
file_path = "data_cuoiky.csv"  # Thay bằng tên file của bạn
data = pd.read_csv(file_path)

# Chuẩn hóa dữ liệu
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Bước 2: Phân tích khoảng cách k-NN
n_neighbors = 10  # Giá trị này thường đặt gần bằng `min_samples`
nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(data_scaled)
distances, indices = nbrs.kneighbors(data_scaled)

# Lấy khoảng cách từ k-NN (sắp xếp giảm dần)
distances = np.sort(distances[:, n_neighbors - 1])  # Khoảng cách lớn nhất của mỗi điểm
plt.figure(figsize=(10, 6))
plt.plot(distances)
plt.title("K-NN Distance Plot")
plt.xlabel("Data Points (sorted by distance)")
plt.ylabel(f"{n_neighbors}-th Nearest Neighbor Distance")
plt.show()

# Giá trị `eps` có thể chọn tại điểm "knee" (uốn cong) trong đồ thị này.
