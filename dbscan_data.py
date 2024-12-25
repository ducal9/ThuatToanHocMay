import pandas as pd
import numpy as np

# Bước 1: Đọc dữ liệu từ file CSV
file_path = 'uncleaned_data.csv'  # Thay bằng đường dẫn file của bạn
df = pd.read_csv(file_path)

# Hiển thị thông tin ban đầu về dữ liệu
print("Dữ liệu ban đầu:")
print(df.info())
print(df.head())

# **Bước 2: Xóa các dòng hoặc cột trùng lặp**
# Loại bỏ các dòng trùng lặp
df = df.drop_duplicates()
print("\nSau khi loại bỏ các dòng trùng lặp:")
print(df.shape)

# **Bước 3: Xử lý giá trị thiếu (NaN)**
# Hiển thị số lượng giá trị thiếu trong mỗi cột
print("\nSố lượng giá trị thiếu trong mỗi cột:")
print(df.isnull().sum())

# Xóa các cột có quá nhiều giá trị thiếu (ví dụ: >50%)
threshold = 0.5  # Ngưỡng (50%)
missing_percent = df.isnull().sum() / len(df)
columns_to_drop = missing_percent[missing_percent > threshold].index
df = df.drop(columns=columns_to_drop)

# Điền giá trị thiếu bằng giá trị trung bình (cho dữ liệu số)
numeric_columns = df.select_dtypes(include=[np.number]).columns
df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())

# Điền giá trị thiếu bằng giá trị phổ biến nhất (cho dữ liệu phân loại)
categorical_columns = df.select_dtypes(include=['object']).columns
for col in categorical_columns:
    df[col] = df[col].fillna(df[col].mode()[0])

print("\nSau khi xử lý giá trị thiếu:")
print(df.isnull().sum())

# **Bước 4: Phát hiện và xử lý giá trị lỗi (outliers)**
# Phát hiện outliers bằng IQR (Interquartile Range)
for col in numeric_columns:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Thay thế các giá trị ngoại lệ bằng NaN
    df[col] = np.where((df[col] < lower_bound) | (df[col] > upper_bound), np.nan, df[col])

# Điền giá trị ngoại lệ đã thay bằng NaN
df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())

print("\nSau khi xử lý outliers:")
print(df.describe())

# **Bước 5: Chuẩn hóa dữ liệu**
# Chuẩn hóa các cột số về khoảng [0, 1] bằng MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

print("\nDữ liệu sau khi chuẩn hóa:")
print(df.head())

# **Bước 6: Xóa khoảng trắng thừa trong các cột văn bản**
for col in categorical_columns:
    df[col] = df[col].str.strip()

# Lưu dữ liệu đã làm sạch vào file mới
df.to_csv('cleaned_data.csv', index=False)
print("\nDữ liệu đã được làm sạch và lưu vào 'cleaned_data.csv'")
