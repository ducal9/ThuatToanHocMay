import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


def generate_sample_data(num_rows=100, num_features=4, seed=None):
    """
    Tạo dữ liệu mẫu ngẫu nhiên với số dòng và số đặc trưng tùy chỉnh.

    Parameters:
        num_rows (int): Số lượng dòng dữ liệu.
        num_features (int): Số lượng đặc trưng.
        seed (int, optional): Giá trị seed để tái lập kết quả.

    Returns:
        pd.DataFrame: DataFrame chứa dữ liệu mẫu.
    """
    if seed is not None:
        np.random.seed(seed)
    
    data = {
        f"Feature_{i+1}": np.random.uniform(0, 10, num_rows)  # Giá trị ngẫu nhiên trong khoảng [0, 10]
        for i in range(num_features)
    }
    return pd.DataFrame(data)

# Gọi hàm để tạo dữ liệu mẫu
sample_data = generate_sample_data(num_rows=100, num_features=4, seed=42)

# Xem trước dữ liệu
print(sample_data.head())

# 1. Đọc dữ liệu

file_path = 'du_lieu.csv'  # Thay bằng đường dẫn file CSV của bạn
data = pd.read_csv(file_path)

# 2. Kiểm tra dữ liệu
print(data.head())
print(data.info())

# 3. Lọc dữ liệu định lượng và chuẩn hóa
numerical_data = data.select_dtypes(include=[np.number])
scaler = StandardScaler()
scaled_data = scaler.fit_transform(numerical_data)

# 4. Xây dựng cây phân cấp
linkage_matrix = linkage(scaled_data, method='ward')  # Hoặc 'single', 'complete', 'average'

# 5. Vẽ biểu đồ dendrogram
plt.figure(figsize=(10, 7))
dendrogram(linkage_matrix)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Sample Index')
plt.ylabel('Distance')
plt.show()

# 6. Cắt cây để tạo cụm
num_clusters = 3  # Thay đổi số cụm mong muốn
clusters = fcluster(linkage_matrix, num_clusters, criterion='maxclust')

# 7. Gắn nhãn cụm vào dữ liệu gốc
data['Cluster'] = clusters
print(data.head())

# 8. Lưu kết quả
data.to_csv('du_lieu_phan_cum.csv', index=False)
