import pandas as pd
import numpy as np

# Tạo dữ liệu giả lập
np.random.seed(42)
data = {
    "Feature1": np.hstack([
        np.random.normal(loc=5, scale=1, size=300),   # Cụm 1
        np.random.normal(loc=15, scale=1.5, size=300), # Cụm 2
        np.random.normal(loc=30, scale=2, size=300)   # Cụm 3
    ]),
    "Feature2": np.hstack([
        np.random.normal(loc=10, scale=1, size=300),
        np.random.normal(loc=25, scale=1.5, size=300),
        np.random.normal(loc=40, scale=2, size=300)
    ]),
    "Feature3": np.hstack([
        np.random.normal(loc=15, scale=1, size=300),
        np.random.normal(loc=35, scale=1.5, size=300),
        np.random.normal(loc=55, scale=2, size=300)
    ]),
    "Feature4": np.hstack([
        np.random.normal(loc= 20 , scale=1, size=300),
        np.random.normal(loc= 50, scale=1.5, size=300),
        np.random.normal(loc= 65, scale=2, size=300)
    ])
}

# Tạo DataFrame
df = pd.DataFrame(data)

# Lưu vào file CSV
file_path = "birch_data.csv"
df.to_csv(file_path, index=False)
print(f"File '{file_path}' đã được tạo.")
