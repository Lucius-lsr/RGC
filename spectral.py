import math

import numpy as np
from sklearn.cluster import KMeans
from rgc import RGC
from data import get_data


def calLapMat(S):
    """
    计算拉普拉斯矩阵
    :param S: adjacent matrix with size n*n
    :return: Laplacian matrix L
    """
    D = np.sum(S, axis=1)
    L = np.diag(D) - S
    sqrtD = np.diag(1.0 / (D ** 0.5))
    L = np.dot(np.dot(sqrtD, L), sqrtD)
    return L


def L2Dist(a, b):
    """
    calculate the L2 distance between a and b
    """
    return np.sqrt(np.sum((a - b) ** 2))


def get_closest_dist(point, centroids):
    """
    :param point: a vector of k dims
    :param centroids: a t * k matrix representing t center points
    :return:
    """
    min_dist = math.inf  # 初始设为无穷大
    k = centroids.shape[0]
    for i in range(k):
        dist = L2Dist(centroids[i, :], point)
        if dist < min_dist:
            min_dist = dist
    return min_dist**2


def kpp_centers(data_set):
    """
    choose k centers from data_set
    """
    n, k = data_set.shape
    first_row = np.random.choice(np.arange(n), size=1, replace=False)
    cluster_centers = data_set[first_row, :]
    d = np.zeros(n)
    for _ in range(1, k):
        total = 0.0
        for i in range(n):
            d[i] = get_closest_dist(data_set[i, :], cluster_centers)  # 与最近一个聚类中心的距离
            total += d[i]
        d = d/total
        # 选出下一个中心
        next_row = np.random.choice(np.arange(n), size=1, replace=False, p=d)
        cluster_centers = np.insert(cluster_centers, cluster_centers.shape[0], data_set[next_row,:], axis=0)
    return cluster_centers


def kmeans(A):
    """
    :param A: data set matrix (n * k), k is cluster number
    :return: k centers and a n * 2 matrix, the first row is the cluster num the point belongs to,
    the second row is the dist to the cluster center
    """
    n, k = A.shape
    clusterResult = np.zeros((n, 2))
    needUpdate = True

    # 初始化簇中心
    centers = kpp_centers(A)

    # 循环更新
    while needUpdate:
        needUpdate = False
        for i in range(n):
            minDist = 9999999.0
            minIdx = -1
            for j in range(k):
                dist = L2Dist(A[i, :], centers[j, :])
                if dist < minDist:
                    minDist = dist
                    minIdx = j
            if clusterResult[i, 0] != minIdx:
                needUpdate = True
                clusterResult[i, :] = minIdx, minDist ** 2
        # 更新簇中心
        for i in range(k):
            clusterPoints = A[np.nonzero(clusterResult[:, 0] == j)[0]]
            centers[j:] = np.mean(clusterPoints, axis=0)
    return centers, clusterResult


def spectral(S, k):
    """
    :param S: data set
    :param k: cluster number
    :return:
    """
    # 计算拉普拉斯矩阵
    L = calLapMat(S)

    # 特征值分解
    lam, H = np.linalg.eig(L)

    # 取特征值最大的前k个特征向量
    H = H[:, np.argsort(lam)]
    H = H[:, :k]

    # 行单位化
    tmp = np.sum(H * H, axis=1).reshape(-1, 1)
    H = H / (tmp ** 0.5)

    # kmeans聚类
    center, result = kmeans(H)

    # skLearn调包结果
    sklResult = KMeans(n_clusters=k).fit(H)

    return result[:, 0].reshape(1, -1)[0, :].astype(np.int)


def evaluate(result, label, k):
    correct = 0
    for i in range(k):
        vote = np.zeros(k)
        idx = np.where(result == i)
        truth = label[idx]
        for j in truth:
            vote[j] += 1
        correct += np.max(vote)
    return correct/len(result)

# load data
x, y, num_train, num_class = get_data()

model = RGC(10, 0.1, 1, 1)
model.graph_construct(x)

S = model.S

result = spectral(S, 3)
print(result)
print(evaluate(result, y, 3))
