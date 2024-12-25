import pandas as pd
import numpy as np

# Tạo dữ liệu ngẫu nhiên
np.random.seed(42)

# Tạo các cụm dữ liệu
data1 = np.random.multivariate_normal([5, 5], [[0.8, 0], [0, 0.8]], 300)   # Cụm 1
data2 = np.random.multivariate_normal([15, 10], [[1.2, 0], [0, 1.2]], 200) # Cụm 2
data3 = np.random.multivariate_normal([25, 5], [[1.5, 0], [0, 1.5]], 150)  # Cụm 3

# Gộp dữ liệu
data = np.vstack([data1, data2, data3])

# Tạo DataFrame và lưu vào CSV
df = pd.DataFrame(data, columns=["Feature1", "Feature2"])
file_path = "sting_data.csv"
df.to_csv(file_path, index=False)
print(f"File '{file_path}' đã được tạo.")
