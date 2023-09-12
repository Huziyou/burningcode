import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.decomposition import PCA

data = pd.read_csv('/home/huziyou/pythonFile/IDL/Assignments_0/train_in - Copy.csv')

pca = PCA(n_components=2)
principal_components = pca.fit_transform(data)

# 输出主成分的方差贡献比例
explained_variance_ratio = pca.explained_variance_ratio_
print("解释的方差贡献比例:", explained_variance_ratio)

# 生成可视化图像
plt.figure(figsize=(8, 6))
plt.scatter(principal_components[:, 0], principal_components[:, 1])
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA Visualization (2 Principal Components)')
plt.show()
plt.savefig('my_plot_2.png')
