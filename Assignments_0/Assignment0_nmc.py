import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pandas as pd
import umap

# 使用Pandas加载CSV文件
data_train_in = pd.read_csv('/home/huziyou/pythonFile/IDL/Assignments_0/train_in - Copy.csv', header=None)
data_test_in = pd.read_csv('/home/huziyou/pythonFile/IDL/Assignments_0/test_in - Copy.csv', header=None)
data_label = pd.read_csv('/home/huziyou/pythonFile/IDL/Assignments_0/train_out - Copy.csv', header=None)
data_test_out = pd.read_csv('/home/huziyou/pythonFile/IDL/Assignments_0/test_out - Copy.csv', header=None).squeeze()

# 将数据转换为NumPy数组
point_cloud_train_in = data_train_in.to_numpy()  # 转换train_in数据
point_cloud_test_in = data_test_in.to_numpy()  # 转换test_in数据

# 创建PCA模型，指定降维后的维度
# pca = PCA(n_components=50)  # 指定降维到50维

# # 创建UMAP对象，设置目标维度为10（或您需要的维度）
# umap_model = umap.UMAP(n_components=80)

# tsne_model = TSNE(n_components=3)

# 拟合PCA模型并进行降维
# reduced_point_cloud_train_in = pca.fit_transform(point_cloud_train_in)
# reduced_point_cloud_test_in = pca.fit_transform(point_cloud_test_in)

# reduced_point_cloud_train_in = umap_model.fit_transform(data_train_in)
# reduced_point_cloud_test_in = umap_model.fit_transform(data_test_in)

# reduced_point_cloud_train_in =  tsne_model.fit_transform(data_train_in)
# reduced_point_cloud_test_in = tsne_model.fit_transform(data_test_in)



def pca(X, n_components=None):
    # 计算均值
    mean = np.mean(X, axis=0)

    # 数据中心化
    centered_data = X - mean

    # 计算协方差矩阵
    covariance_matrix = np.cov(centered_data, rowvar=False)

    # 特征值分解
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

    # 对特征向量按特征值降序排序
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, sorted_indices]
    eigenvalues = eigenvalues[sorted_indices]

    # 选择要保留的主成分数量
    if n_components is not None:
        selected_eigenvectors = eigenvectors[:, :n_components]
    else:
        selected_eigenvectors = eigenvectors

    # 将数据投影到选定的主成分上
    reduced_data = np.dot(centered_data, selected_eigenvectors)

    return reduced_data

reduced_point_cloud_train_in = pca(point_cloud_train_in, n_components=50)
reduced_point_cloud_test_in = pca(point_cloud_test_in, n_components=50)

# 选择不降维数据
# reduced_point_cloud_train_in = data_train_in
# reduced_point_cloud_test_in = data_test_in

# 创建包含降维后数据的DataFrame
reduced_data_train_in = pd.DataFrame(data=reduced_point_cloud_train_in)  # 根据实际降维维度进行命名
reduced_data_test_in = pd.DataFrame(data=reduced_point_cloud_test_in)


save_path_train_in = '/home/huziyou/pythonFile/IDL/Assignments_0/reduced_train_in.csv'
save_path_test_in = '/home/huziyou/pythonFile/IDL/Assignments_0/reduced_test_in.csv'

# 将降维后的数据保存为CSV文件
reduced_data_train_in.to_csv(save_path_train_in, index=False)
reduced_data_test_in.to_csv(save_path_test_in, index=False)

#将降维后的train_in数据与标签合并
combined_data = pd.concat([reduced_data_train_in, data_label], axis=1, ignore_index=True) # ignore_index=True可以生成新的连续索引!!!
combined_data.to_csv('/home/huziyou/pythonFile/IDL/Assignments_0/combine_data.csv', index=False, header=False) # header参数决定数据是否包含列名

# 获取数据中的标签列
labels = combined_data.iloc[:, -1]

# 获取唯一的标签值
unique_labels = labels.unique()

# 创建一个字典来存储每个类别的数据和中心值
class_data = {}
class_centers = {}

"""
完成了训练步骤
因为处理完了所有的训练数据
"""
# 遍历每个唯一的标签值
for label in unique_labels:
    # 选择属于当前标签的数据
    class_data[label] = combined_data[labels == label].iloc[:, :-1]  # 去除最后一列作为数据
    # 计算当前类别的数据中心值
    class_centers[label] = class_data[label].mean()


def nearest_mean_classifier(train_data, test_data, class_centers, true_labels_series):
    """
    最近均值分类器函数
    
    参数：
    - train_data: 训练集数据，包含特征和标签
    - test_data: 测试集数据，包含特征
    - class_centers: 每个类别的中心值字典
    - true_labels_series: 包含测试集样本真实标签的 Pandas Series
    
    返回：
    - predictions: 测试集样本的预测类别列表
    - class_accuracies: 每个类别的准确度字典
    """
    predictions = []
    
    # 初始化一个字典来存储每个类别的统计信息
    class_stats = {label: {'correct': 0, 'total': 0} for label in class_centers.keys()}
    
    # 遍历测试集中的每个样本
    for _, row in test_data.iterrows():
        sample = np.array(row)  # 将样本数据转换为 NumPy 数组
        
        # 初始化变量来存储距离最近的类别和距离
        nearest_class = None
        nearest_distance = float('inf')  # 初始化为正无穷
        
        # 计算样本与每个类别中心的距离，选择距离最近的类别
        for label, center in class_centers.items():
            distance = np.linalg.norm(sample - center)  # 使用欧氏距离计算距离
            if distance < nearest_distance:
                nearest_distance = distance
                nearest_class = label
        
        # 将样本分配给距离最近的类别
        predictions.append(nearest_class)
        
        # 获取样本的真实标签
        true_label = true_labels_series.iloc[row.name]
        
        # 更新类别统计信息
        class_stats[true_label]['total'] += 1
        if nearest_class == true_label:
            class_stats[true_label]['correct'] += 1
    
    # 计算每个类别的准确度
    class_accuracies = {}
    for label, stats in class_stats.items():
        accuracy = stats['correct'] / stats['total'] if stats['total'] > 0 else 0.0
        class_accuracies[label] = accuracy
    
    # 打印各个类别的分类准确度
    for label, accuracy in class_accuracies.items():
        print(f'Class {label}: Accuracy = {accuracy:.2f}')
    
    return predictions, class_accuracies

# 调用分类器函数

predictions, class_accuracies = nearest_mean_classifier(reduced_data_train_in, reduced_data_test_in, class_centers, data_test_out)

