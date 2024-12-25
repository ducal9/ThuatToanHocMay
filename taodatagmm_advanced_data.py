import pandas as pd
import numpy as np

# Đặt seed để đảm bảo tính tái lập
np.random.seed(42)

# Tạo các cụm phân phối Gaussian nhiều chiều
data1 = np.random.multivariate_normal(
    mean=[5, 5, 10, 0, 15], cov=[[2, 0.5, 0.3, 0.2, 0.1],
                                 [0.5, 2, 0.4, 0.3, 0.2],
                                 [0.3, 0.4, 1.5, 0.2, 0.1],
                                 [0.2, 0.3, 0.2, 1.2, 0.1],
                                 [0.1, 0.2, 0.1, 0.1, 1.0]], size=300)

data2 = np.random.multivariate_normal(
    mean=[20, 25, 5, -5, 10], cov=[[3, 0.2, 0.1, 0.1, 0.1],
                                   [0.2, 3, 0.1, 0.1, 0.1],
                                   [0.1, 0.1, 2, 0.1, 0.1],
                                   [0.1, 0.1, 0.1, 2, 0.1],
                                   [0.1, 0.1, 0.1, 0.1, 1.5]], size=300)

data3 = np.random.multivariate_normal(
    mean=[10, 15, -5, 5, -10], cov=[[2.5, -0.3, 0.1, 0.1, -0.2],
                                    [-0.3, 2.5, 0.1, -0.2, 0.1],
                                    [0.1, 0.1, 2, 0.1, 0.1],
                                    [0.1, -0.2, 0.1, 1.5, 0.1],
                                    [-0.2, 0.1, 0.1, 0.1, 1.8]], size=300)

# Kết hợp các cụm
data = np.vstack([data1, data2, data3])

# Tạo DataFrame với 5 đặc trưng
df = pd.DataFrame(data, columns=["Feature1", "Feature2", "Feature3", "Feature4", "Feature5"])

# Lưu vào file CSV
file_path = "gmm_advanced_data.csv"
df.to_csv(file_path, index=False)
print(f"File '{file_path}' đã được tạo.")
