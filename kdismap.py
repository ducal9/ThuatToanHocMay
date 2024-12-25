import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

def plot_k_distance(X, k=4):
    """
    Vẽ biểu đồ K-Distance để xác định giá trị epsilon cho DBSCAN.

    Parameters:
    - X: ndarray, dữ liệu đầu vào (mảng các điểm).
    - k: int, số lượng hàng xóm (k-1) để tính khoảng cách.
    """
    # Tính khoảng cách đến hàng xóm thứ k
    nbrs = NearestNeighbors(n_neighbors=k).fit(X)
    distances, _ = nbrs.kneighbors(X)
    k_distances = np.sort(distances[:, k-1])  # Lấy khoảng cách tới hàng xóm thứ k
    
    # Vẽ biểu đồ
    plt.figure(figsize=(8, 6))
    plt.plot(k_distances, label=f'{k}-distance graph')
    plt.xlabel('Points sorted by distance', fontsize=12)
    plt.ylabel(f'Distance to {k}th nearest neighbor', fontsize=12)
    plt.title(f'K-Distance Graph (k={k})', fontsize=14)
    plt.grid()
    plt.legend()
    plt.show()
from sklearn.datasets import make_blobs

# Tạo dữ liệu giả lập
X, _ = make_blobs(n_samples=300, centers=5, cluster_std=0.5, random_state=1)

# Vẽ biểu đồ k-distance với k=4
plot_k_distance(X, k=4)
