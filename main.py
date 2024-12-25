# Import các thư viện cần thiết
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_blobs

# Tạo dữ liệu mẫu (dữ liệu giả lập cho mục đích ví dụ)
# Sử dụng make_blobs để tạo ra các điểm dữ liệu phân nhóm 
X, _ = make_blobs(n_samples=300, centers=5, cluster_std=0.5, random_state=1)
print(X[:100])
# show dữ liệu mẫu trên terminal



# viet function tính khoảng cách 2 điểm
def distance(p1, p2):
    return np.sqrt(np.sum((p1 - p2) ** 2))

def group_points(points, eps):
    clusters = []  # Danh sách các cụm
    visited = [False] * len(points)  # Mảng đánh dấu các điểm đã được duyệt
     
    # Hàm đệ quy để tìm tất cả các điểm liên kết với điểm hiện tại
    def find_neighbours(i):
        neighbours = []
        for j, other_point in enumerate(points):
            if not visited[j] and distance(points[i], other_point) < eps:
                neighbours.append(j)
        return neighbours

    for i in range(len(points)):
        if visited[i]:
            continue
        
        # Khởi tạo một cụm mới
        cluster = [i]
        visited[i] = True
        
        # Duyệt qua các điểm trong cụm và tìm điểm liên kết tim hang xom 
        neighbours = find_neighbours(i)
        while neighbours:
            current = neighbours.pop()
            if not visited[current]:
                visited[current] = True
                cluster.append(current)
                neighbours.extend(find_neighbours(current))
        
        clusters.append([points[i] for i in cluster])  # Thêm cụm vào danh sách

    return clusters

eps = 0.6
min_point = 4

# lặp mảng X để tìm ra các điểm lõi 
core_points = []
for i in range(len(X)):
    count = 0
    for j in range(len(X)):
        if distance(X[i], X[j]) <= eps:
            count += 1
    if count >= min_point:
        core_points.append(X[i])
# Nhóm các điểm lõi lại thành các cụm 
clusters = group_points(core_points, eps)

y_dbscan = np.ones(len(X)) * -1

for i in range(len(X)):
    for j in range(len(clusters)):
        found = False
        for k in range(len(clusters[j])):
            if distance(X[i], clusters[j][k]) <= eps:
                y_dbscan[i] = j
                found = True
                break
        if found:
            break




# Trực quan hóa kết quả phân cụm
plt.figure(figsize=(8, 6))

# Lặp qua các giá trị phân nhóm và vẽ các điểm với màu sắc khác nhau
plt.scatter(X[:, 0], X[:, 1], c=y_dbscan, cmap='Paired', marker='o')
plt.title("DBSCAN Clustering")
plt.xlabel("X1")
plt.ylabel("X2")
plt.colorbar(label='Cluster ID')

# Hiển thị đồ thị
plt.show()
