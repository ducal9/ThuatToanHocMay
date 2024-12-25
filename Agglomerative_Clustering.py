import pandas as pd
import numpy as np

# Tạo dữ liệu ngẫu nhiên
np.random.seed(42)
data = {
    "Feature1": np.random.uniform(10, 20, 1000),  # Giá trị ngẫu nhiên từ 10 đến 20
    "Feature2": np.random.uniform(5, 15, 1000),   # Giá trị ngẫu nhiên từ 5 đến 15
    "Feature3": np.random.uniform(1, 10, 1000),   
    "Feature4": np.random.uniform(0 , 1, 1000)
}

# Tạo DataFrame
df = pd.DataFrame(data)

# Lưu thành file CSV
file_path = "agglomerative_data.csv"
df.to_csv(file_path, index=False)
print(f"File '{file_path}' đã được tạo.")
