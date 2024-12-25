
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
import numpy as np

from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
import pandas as pd 
import matplotlib.pyplot as plt

# Bước 1: Định nghĩa các giá trị eps và min_samples để thử nghiệm
eps = 3  # Thay bằng giá trị từ đồ thị K-NN
min_samples_range = range(2, 19)  # Giá trị min_samples cần thử nghiệm

# Bước 2: Lặp qua các giá trị và tính Silhouette Score
best_score = -1
best_params = None
df = pd.read_csv('data_cuoiky.csv')
from sklearn.preprocessing import StandardScaler
#cho nhữ điểm giữ liệu vd quấ lớn thid cho nó nhỏ lại 
scaler = StandardScaler()
scaler_X = scaler.fit_transform(df)


for min_samples in min_samples_range:
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(scaler_X)
    
    # Loại bỏ trường hợp chỉ có 1 cụm hoặc toàn bộ là nhiễu
    if len(set(labels)) > 1 and -1 not in set(labels):
        score = silhouette_score(scaler_X, labels)
        print(f"eps={eps}, min_samples={min_samples}, Silhouette Score={score}")
        # Đảm bảo có ít nhất 2 cụm
        if score > best_score:
            best_score = score
            best_params = (eps, min_samples)

# In ra kết quả tốt nhất
if best_params:
    print(f"Best params: eps={best_params[0]}, min_samples={best_params[1]}, Silhouette Score={best_score}")
else:
    print("No valid clustering found.")
