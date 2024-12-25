from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Bước 1: Đọc dữ liệu từ file CSV
file_path = "data_cuoiky.csv"
data = pd.read_csv(file_path)

# Bước 2: Áp dụng Gaussian Mixture Model (GMM) để thực hiện EM
n_components = 3  # Số cụm
gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42)
gmm.fit(data)

# Gắn nhãn cụm vào dữ liệu
data['Cluster'] = gmm.predict(data)

# Bước 3: Đánh giá xác suất của từng điểm thuộc về cụm
data['Probability'] = gmm.predict_proba(data.iloc[:, :-1]).max(axis=1)

# Bước 4: Giảm chiều dữ liệu bằng PCA để trực quan hóa
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(data.iloc[:, :-2])  # Bỏ cột 'Cluster' và 'Probability'

# Trực quan hóa cụm
plt.figure(figsize=(10, 8))
sns.scatterplot(x=reduced_data[:, 0], y=reduced_data[:, 1], hue=data['Cluster'], palette='Set2', s=50, alpha=0.8)
plt.title("EM Algorithm (GMM) - Clustering Visualization")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend(title="Cluster")
plt.show()

# Bước 5: Lưu kết quả phân cụm vào file CSV
output_file = "em_advanced_clusters.csv"
data.to_csv(output_file, index=False)
print(f"Kết quả phân cụm đã được lưu vào '{output_file}'")
