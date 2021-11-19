import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# 基于特征值分解协方差矩阵实现PCA算法

# 数据集装载，X为13个特征

url = "wine.csv"
dataset = pd.read_csv(url, names=None)
X = dataset.iloc[:,1:].values

print("原始数据 178X13 ：")
print(X)

# 去平均值，即去中心化，每个特征减去平均值

data = X - np.mean(X, axis = 0, keepdims = True)


# 计算协方差矩阵

cov = np.dot(data.T, data)


# 求特征值与特征向量

eig_values, eig_vector = np.linalg.eig(cov)


#  将特征值按照从大到小排序，选取其中最大的2(降到2维)个

indexs_ = np.argsort(-eig_values)[:2]


# 特征向量作为行向量组成特征向量矩阵

picked_eig_vector = eig_vector[:, indexs_]


# 数据降维到二维

data_2dim = np.dot(data, picked_eig_vector)
print("降维后的数据 178X2：")
print(data_2dim)


# 数据恢复

data_approx=np.dot(data_2dim,picked_eig_vector.T)
X_approx=data_approx+np.mean(X, axis = 0, keepdims = True)
print("复原的数据 178X13 ：")
print(X_approx)


# 计算准确度（百分误差）

print("单个数据准确度：")
accuracy=(abs(X_approx-X))/X
print(accuracy)

print("每组数据准确度：")
accuracy_row=accuracy.sum (axis=1)/13
print(accuracy_row)

print("平均准确度:")
accuracy_avg=accuracy.sum()/(150*13)
print(accuracy_avg)


# 可视化
'''
# 绘制散点图

plt.figure(figsize=(8,6))
plt.title("PCA_dimmensionality_reduction")
plt.scatter(data_2dim[:, 0], data_2dim[:, 1], c = Y)
plt.show()

'''
